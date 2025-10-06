#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pg_list.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/Column.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace mlir::relalg {
class CountRowsOp;
class BaseTableOp;
} // namespace mlir::relalg

namespace {
using namespace pgx_lower::frontend::sql;
using namespace postgresql_ast;

auto find_all_aggrefs(Expr* expr, std::vector<Aggref*>& result) -> void {
    if (!expr)
        return;
    if (IsA(expr, Aggref)) {
        result.push_back(reinterpret_cast<Aggref*>(expr));
        return;
    }

    if (IsA(expr, OpExpr)) {
        const auto* op_expr = reinterpret_cast<OpExpr*>(expr);
        ListCell* lc;
        foreach (lc, op_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, FuncExpr)) {
        const auto* func_expr = reinterpret_cast<FuncExpr*>(expr);
        ListCell* lc;
        foreach (lc, func_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, BoolExpr)) {
        const auto* bool_expr = reinterpret_cast<BoolExpr*>(expr);
        ListCell* lc;
        foreach (lc, bool_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }
}

auto getAggregateFunction(const std::string& funcName) -> mlir::relalg::AggrFunc {
    return (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
           : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
           : (funcName == "min") ? mlir::relalg::AggrFunc::min
           : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                 : mlir::relalg::AggrFunc::count;
}

auto getFirstAggregateArgument(const Aggref* aggref) -> TargetEntry* {
    if (!aggref->args || list_length(aggref->args) == 0)
        return nullptr;
    auto* argTE = static_cast<TargetEntry*>(linitial(aggref->args));
    return (argTE && argTE->expr) ? argTE : nullptr;
}

auto createColumnDef(mlir::relalg::ColumnManager& columnManager, const std::string& scopeName,
                     const std::string& columnName, const mlir::Type columnType) -> mlir::relalg::ColumnDefAttr {
    const auto colDef = columnManager.createDef(scopeName, columnName);
    colDef.getColumn().type = columnType;
    return colDef;
}

auto processCountStarAggregate(mlir::OpBuilder& aggr_builder, const mlir::Location loc, mlir::Value relation,
                               mlir::relalg::ColumnDefAttr& attrDef, mlir::relalg::ColumnManager& columnManager,
                               const std::string& aggrScopeName, const std::string& aggColumnName,
                               std::map<int, std::pair<std::string, std::string>>& aggregateMappings, const int aggno,
                               const bool logDebug = true) -> mlir::Value {
    if (logDebug) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Processing COUNT(*) aggregate aggno=%d", aggno);
    }

    auto i64Type = mlir::IntegerType::get(aggr_builder.getContext(), 64);
    attrDef = createColumnDef(columnManager, aggrScopeName, aggColumnName, i64Type);

    aggregateMappings[aggno] = std::make_pair(aggrScopeName, aggColumnName);

    if (logDebug) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Added COUNT(*) mapping aggno=%d -> (%s, %s)", aggno, aggrScopeName.c_str(),
                aggColumnName.c_str());
    }

    return aggr_builder.create<mlir::relalg::CountRowsOp>(loc, i64Type, relation);
}

auto createAggregateOperation(mlir::OpBuilder& aggr_builder, const mlir::Location loc, mlir::Type resultType,
                              mlir::relalg::AggrFunc aggrFuncEnum, mlir::Value relation,
                              mlir::relalg::ColumnRefAttr columnRef, const bool isDistinct) -> mlir::Value {
    if (isDistinct) {
        auto distinctStream = aggr_builder.create<mlir::relalg::ProjectionOp>(
            loc, mlir::relalg::TupleStreamType::get(aggr_builder.getContext()),
            mlir::relalg::SetSemanticAttr::get(aggr_builder.getContext(), mlir::relalg::SetSemantic::distinct),
            relation, aggr_builder.getArrayAttr({columnRef}));

        return aggr_builder.create<mlir::relalg::AggrFuncOp>(loc, resultType, aggrFuncEnum, distinctStream.getResult(),
                                                             columnRef);
    } else {
        return aggr_builder.create<mlir::relalg::AggrFuncOp>(loc, resultType, aggrFuncEnum, relation, columnRef);
    }
}

} // namespace

namespace postgresql_ast {

using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLASTTranslator::Impl::translate_agg(QueryCtxT& ctx, const Agg* agg) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!agg || !agg->plan.lefttree)
        throw std::runtime_error("invalid input");

    auto childResult = translate_plan_node(ctx, agg->plan.lefttree);
    if (!childResult.op || !childResult.op->getNumResults())
        throw std::runtime_error("Failed to translate Agg child plan");

    auto childOutput = childResult.op->getResult(0);
    auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    auto groupByAttrs = std::vector<mlir::Attribute>{};
    const auto type_mapper = PostgreSQLTypeMapper(*ctx.builder.getContext());

    // Section 1: Identify groups - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    {
        if (agg->numCols > 0 && agg->grpColIdx) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Building GROUP BY from grpColIdx, numCols=%d", agg->numCols);
            for (int i = 0; i < agg->numCols; i++) {
                int colIdx = agg->grpColIdx[i];
                if (colIdx > 0 && colIdx <= static_cast<int>(childResult.columns.size())) {
                    const auto& childCol = childResult.columns[colIdx - 1];
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: GROUP BY column %d: table='%s' name='%s' (from child column at index %d)", i,
                            childCol.table_name.c_str(), childCol.column_name.c_str(), colIdx - 1);
                    auto colRef = columnManager.createRef(childCol.table_name, childCol.column_name);
                    colRef.getColumn().type = childCol.mlir_type;
                    groupByAttrs.push_back(colRef);
                }
            }
        }

        // Also scan targetlist for additional GROUP BY columns that PostgreSQL optimized out of grpColIdx
        // (e.g., when grouping by PK + dependent column, PostgreSQL may only list PK in grpColIdx)
        // IMPORTANT: Only add if the column exists in child result (prevents nested aggregation issues)
        if (agg->plan.targetlist) {
            ListCell* lc;
            foreach (lc, agg->plan.targetlist) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && tle->ressortgroupref > 0 && !tle->resjunk && IsA(tle->expr, Var)) {
                    auto* var = reinterpret_cast<Var*>(tle->expr);

                    if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                        const auto& childCol = childResult.columns[var->varattno - 1];

                        // Check if column already in GROUP BY
                        bool alreadyInGroup = false;
                        for (const auto& attr : groupByAttrs) {
                            if (auto existingColRef = mlir::dyn_cast<mlir::relalg::ColumnRefAttr>(attr)) {
                                auto existingName = existingColRef.getName();
                                if (existingName.getRootReference().str() == childCol.table_name
                                    && existingName.getLeafReference().str() == childCol.column_name)
                                {
                                    alreadyInGroup = true;
                                    break;
                                }
                            }
                        }

                        // GUARD: Only add if column exists in child output
                        // This prevents issues in nested aggregations where outer agg references columns
                        // from inner agg's child that inner agg didn't output
                        if (!alreadyInGroup) {
                            // Check if this column was actually output by the child
                            bool existsInChild = false;
                            for (const auto& col : childResult.columns) {
                                if (col.table_name == childCol.table_name && col.column_name == childCol.column_name) {
                                    existsInChild = true;
                                    break;
                                }
                            }

                            if (existsInChild) {
                                PGX_LOG(AST_TRANSLATE, DEBUG,
                                        "Agg: Adding GROUP BY column from targetlist: %s.%s (ressortgroupref=%d)",
                                        childCol.table_name.c_str(), childCol.column_name.c_str(), tle->ressortgroupref);
                                auto colRef = columnManager.createRef(childCol.table_name, childCol.column_name);
                                colRef.getColumn().type = childCol.mlir_type;
                                groupByAttrs.push_back(colRef);
                            }
                        }
                    }
                }
            }
        }
    }

    static size_t aggrId = 0;
    auto aggrScopeName = "aggr" + std::to_string(aggrId++);
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    if (!agg->plan.targetlist || agg->plan.targetlist->length <= 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning childResult with %zu columns",
                childResult.columns.size());
        return childResult;
    }
    auto* block = new mlir::Block;
    block->addArgument(tupleStreamType, ctx.builder.getUnknownLoc());
    block->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

    mlir::OpBuilder aggr_builder(ctx.builder.getContext());
    aggr_builder.setInsertionPointToStart(block);

    auto createdValues = std::vector<mlir::Value>{};
    auto createdCols = std::vector<mlir::Attribute>{};

    auto aggregateMappings = std::map<int, std::pair<std::string, std::string>>();

    auto needs_post_processing = std::set<int>();
    auto post_process_exprs = std::map<int, Expr*>();

    auto process_single_aggregate = [&](const Aggref* aggref) -> void {
        char* rawFuncName = get_func_name(aggref->aggfnoid);
        if (!rawFuncName) {
            PGX_WARNING("Failed to find a function name!");
            return;
        }
        const auto funcName = std::string(rawFuncName);
        pfree(rawFuncName);

        auto aggColumnName = "agg_" + std::to_string(aggref->aggno);
        const auto relation = block->getArgument(0);
        mlir::Value aggResult;

        if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
            // COUNT(*) special case
            mlir::relalg::ColumnDefAttr attrDef;
            aggResult = processCountStarAggregate(aggr_builder, ctx.builder.getUnknownLoc(), relation, attrDef,
                                                  columnManager, aggrScopeName, aggColumnName, aggregateMappings,
                                                  aggref->aggno);
            createdCols.push_back(attrDef);
            createdValues.push_back(aggResult);
        } else {
            const auto argTE = getFirstAggregateArgument(aggref);
            if (!argTE)
                return;

            const auto childCtx = QueryCtxT::createChildContextWithOuter(ctx, childResult);

            TranslationResult exprContext = childResult;
            exprContext.op = childOutput.getDefiningOp();
            auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                childCtx, argTE->expr, exprContext, "agg_expr_" + std::to_string(aggref->aggno));

            if (stream != childOutput) {
                childOutput = llvm::cast<mlir::OpResult>(stream);
                const mlir::Type actual_type = column_ref.getColumn().type;
                const bool is_nullable = mlir::isa<mlir::db::NullableType>(actual_type);

                childResult.columns.push_back({.table_name = table_name,
                                               .column_name = column_name,
                                               .type_oid = exprType(reinterpret_cast<Node*>(argTE->expr)),
                                               .typmod = exprTypmod(reinterpret_cast<Node*>(argTE->expr)),
                                               .mlir_type = actual_type,
                                               .nullable = is_nullable});
            }

            mlir::Type resultType;
            if (funcName == "count") {
                resultType = ctx.builder.getI64Type();
            } else if (aggref->aggtype == 17 && aggref->aggargtypes && list_length(aggref->aggargtypes) > 0) {
                // BYTEAOID (17) indicates PostgreSQL is using polymorphic aggregate with internal state
                // Use the actual argument type for result type (works for SUM/MIN/MAX)
                Oid argTypeOid = lfirst_oid(list_head(aggref->aggargtypes));
                resultType = type_mapper.map_postgre_sqltype(argTypeOid, -1, true);
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "Polymorphic aggregate: using aggargtypes for result type: aggtype=%u -> argtype=%u",
                        aggref->aggtype, argTypeOid);
            } else {
                resultType = type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);
            }
            const auto attrDef = createColumnDef(columnManager, aggrScopeName, aggColumnName, resultType);
            aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);

            const auto aggrFuncEnum = getAggregateFunction(funcName);
            aggResult = createAggregateOperation(aggr_builder, ctx.builder.getUnknownLoc(), resultType, aggrFuncEnum,
                                                 relation, column_ref, aggref->aggdistinct);
            createdCols.push_back(attrDef);
            createdValues.push_back(aggResult);
        }
    };

    // Section 2: Translate expressions in the targetlist - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    {
        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr)
                continue;

            if (IsA(te->expr, Aggref)) {
                // Direct aggregate: SUM(x), COUNT(*), etc.
                auto aggref = reinterpret_cast<Aggref*>(te->expr);
                char* rawFuncName = get_func_name(aggref->aggfnoid);
                if (!rawFuncName)
                    continue;
                auto funcName = std::string(rawFuncName);
                pfree(rawFuncName);

                auto aggColumnName = te->resname ? te->resname : funcName + "_" + std::to_string(aggref->aggno);
                auto relation = block->getArgument(0);
                mlir::Value aggResult;

                if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
                    mlir::relalg::ColumnDefAttr attrDef;
                    aggResult = processCountStarAggregate(aggr_builder, ctx.builder.getUnknownLoc(), relation, attrDef,
                                                          columnManager, aggrScopeName, aggColumnName,
                                                          aggregateMappings, aggref->aggno);
                    createdCols.push_back(attrDef);
                    createdValues.push_back(aggResult);
                } else {
                    auto argTE = getFirstAggregateArgument(aggref);
                    if (!argTE)
                        continue;

                    const auto childCtx = QueryCtxT::createChildContextWithOuter(ctx, childResult);
                    TranslationResult exprContext = childResult;
                    exprContext.op = childOutput.getDefiningOp();
                    auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                        childCtx, argTE->expr, exprContext, "agg_expr_" + std::to_string(aggref->aggno));

                    if (stream != childOutput) {
                        childOutput = llvm::cast<mlir::OpResult>(stream);
                        // Extract the actual MLIR type from the column reference attribute
                        mlir::Type actual_type = column_ref.getColumn().type;
                        bool is_nullable = mlir::isa<mlir::db::NullableType>(actual_type);

                        childResult.columns.push_back({.table_name = table_name,
                                                       .column_name = column_name,
                                                       .type_oid = exprType(reinterpret_cast<Node*>(argTE->expr)),
                                                       .typmod = exprTypmod(reinterpret_cast<Node*>(argTE->expr)),
                                                       .mlir_type = actual_type,
                                                       .nullable = is_nullable});
                    }

                    mlir::Type resultType;
                    if (funcName == "count") {
                        resultType = ctx.builder.getI64Type();
                    } else if (aggref->aggtype == 17 && aggref->aggargtypes && list_length(aggref->aggargtypes) > 0) {
                        // BYTEAOID (17) indicates PostgreSQL is using polymorphic aggregate with internal state
                        // Use the actual argument type for result type (works for SUM/MIN/MAX)
                        Oid argTypeOid = lfirst_oid(list_head(aggref->aggargtypes));
                        resultType = type_mapper.map_postgre_sqltype(argTypeOid, -1, true);
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "Polymorphic aggregate: using aggargtypes for result type: aggtype=%u -> argtype=%u",
                                aggref->aggtype, argTypeOid);
                    } else {
                        resultType = type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);
                    }
                    auto attrDef = createColumnDef(columnManager, aggrScopeName, aggColumnName, resultType);
                    aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);

                    auto aggrFuncEnum = getAggregateFunction(funcName);
                    aggResult = createAggregateOperation(aggr_builder, ctx.builder.getUnknownLoc(), resultType,
                                                         aggrFuncEnum, relation, column_ref, aggref->aggdistinct);
                    createdCols.push_back(attrDef);
                    createdValues.push_back(aggResult);
                }
            } else {
                // This could be a complex expression with nested aggregation, like SUM(x) / SUM(y) has two aggregations
                // inside of it.
                auto nested_aggrefs = std::vector<Aggref*>();
                find_all_aggrefs(te->expr, nested_aggrefs);

                if (!nested_aggrefs.empty()) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu nested Aggrefs in complex expression at resno=%d",
                            nested_aggrefs.size(), te->resno);

                    for (auto* nested_aggref : nested_aggrefs) {
                        if (!aggregateMappings.contains(nested_aggref->aggno))
                            process_single_aggregate(nested_aggref);
                    }

                    needs_post_processing.insert(te->resno);
                    post_process_exprs[te->resno] = te->expr;

                    PGX_LOG(AST_TRANSLATE, DEBUG, "Marked resno=%d for post-processing (full expr with %zu aggregates)",
                            te->resno, nested_aggrefs.size());
                }
            }
        }
    }

    // Section 3: HAVING clause that isn't in the target list - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    {
        if (agg->plan.qual && agg->plan.qual->length > 0) {
            auto having_aggrefs = std::vector<Aggref*>();
            ListCell* qual_lc;
            foreach (qual_lc, agg->plan.qual) {
                auto* qual_expr = static_cast<Expr*>(lfirst(qual_lc));
                find_all_aggrefs(qual_expr, having_aggrefs);
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu aggregate(s) in HAVING clause", having_aggrefs.size());
            for (auto* aggref : having_aggrefs) {
                if (aggregateMappings.contains(aggref->aggno)) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Aggregate aggno=%d already in mappings, skipping", aggref->aggno);
                } else {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING aggregate aggno=%d", aggref->aggno);
                    process_single_aggregate(aggref);
                }
            }
        }
    }

    aggr_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), createdValues);
    auto aggOp = ctx.builder.create<mlir::relalg::AggregationOp>(ctx.builder.getUnknownLoc(), tupleStreamType,
                                                                 childOutput, ctx.builder.getArrayAttr(groupByAttrs),
                                                                 ctx.builder.getArrayAttr(createdCols));
    aggOp.getAggrFunc().push_back(block);

    mlir::Value finalOutput = aggOp;
    auto finalScope = aggrScopeName;

    // Section 4: Post processing - evaluate outer expressions that had an aggregation inside of them - - - - - - - - -
    // -
    {
        if (!needs_post_processing.empty()) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating post-processing MapOp for %zu expressions",
                    needs_post_processing.size());

            auto postMapScope = columnManager.getUniqueScope("postmap");
            std::vector<mlir::relalg::ColumnDefAttr> postMapCols;

            auto mapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), aggOp,
                                                                 ctx.builder.getArrayAttr({}));

            auto& mapRegion = mapOp.getPredicate();
            auto* mapBlock = new mlir::Block;
            mapRegion.push_back(mapBlock);
            mapBlock->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

            auto mapBuilder = mlir::OpBuilder(mapBlock, mapBlock->begin());
            auto mapValues = std::vector<mlir::Value>();

            auto postProcResult = TranslationResult();
            postProcResult.op = aggOp.getOperation();
            postProcResult.current_scope = aggrScopeName;
            std::map<std::pair<int, int>, std::pair<std::string, std::string>> agg_mappings;
            for (const auto& [aggno, mapping] : aggregateMappings) {
                agg_mappings[{OUTER_VAR, aggno}] = mapping;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping for post-processing: aggno=%d -> (%s, %s)",
                        aggno, mapping.first.c_str(), mapping.second.c_str());
            }

            for (const auto resno : needs_post_processing) {
                auto* full_expr = post_process_exprs[resno];

                PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing resno=%d with expression type=%d", resno, full_expr->type);

                auto postCtx = create_child_context_with_var_mappings(
                    QueryCtxT::createChildContext(ctx, mapBuilder, mapBlock->getArgument(0)), agg_mappings);
                auto post_value = translate_expression(postCtx, full_expr);

                auto colName = "postproc_" + std::to_string(resno);
                auto colDef = columnManager.createDef(postMapScope, colName);
                colDef.getColumn().type = post_value.getType();

                postMapCols.push_back(colDef);
                mapValues.push_back(post_value);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Created post-processing column: %s.%s", postMapScope.c_str(),
                        colName.c_str());
            }

            mapBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mapValues);

            auto postMapAttrs = std::vector<mlir::Attribute>();
            for (const auto& col : postMapCols) {
                postMapAttrs.push_back(col);
            }
            mapOp.setComputedColsAttr(ctx.builder.getArrayAttr(postMapAttrs));

            finalOutput = mapOp;
            finalScope = postMapScope;

            PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing MapOp created with %zu columns", postMapCols.size());
        }
    }

    // Section 5: Build output schema - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    TranslationResult result;
    {
        result.op = finalOutput.getDefiningOp();
        result.current_scope = finalScope;

        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr)
                continue;

            if (IsA(te->expr, Aggref)) {
                auto* aggref = reinterpret_cast<Aggref*>(te->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "Second loop: Processing aggregate aggno=%d", aggref->aggno);
                auto resultType = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)
                                      ? ctx.builder.getI64Type()
                                      : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                std::string resultColumnName;
                if (aggregateMappings.contains(aggref->aggno)) {
                    const auto& mapping = aggregateMappings[aggref->aggno];
                    resultColumnName = mapping.second;
                    ctx.varno_resolution[std::make_pair(-2, aggref->aggno)] = mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Added aggregate mapping to TranslationResult: varno=-2, aggno=%d -> (%s, %s)",
                            aggref->aggno, mapping.first.c_str(), mapping.second.c_str());
                } else {
                    PGX_ERROR("Aggregate aggno=%d not found in aggregateMappings", aggref->aggno);
                    throw std::runtime_error("read logs");
                }

                PGX_LOG(AST_TRANSLATE, DEBUG, "About to push_back aggregate column: table='%s', column='%s', type_oid=%u",
                        aggrScopeName.c_str(), resultColumnName.c_str(), aggref->aggtype);
                result.columns.push_back({.table_name = aggrScopeName,
                                          .column_name = resultColumnName,
                                          .type_oid = aggref->aggtype,
                                          .typmod = -1,
                                          .mlir_type = resultType,
                                          .nullable = true});
                PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully pushed aggregate column, result now has %zu columns",
                        result.columns.size());
            } else if (IsA(te->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(te->expr);
                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    const auto& childCol = childResult.columns[var->varattno - 1];
                    result.columns.push_back(childCol);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Adding GROUP BY column '%s' to output (varattno=%d, resname=%s)",
                            childCol.column_name.c_str(), var->varattno, te->resname ? te->resname : "<null>");
                }
            } else {
                Oid exprTypeOid = exprType(reinterpret_cast<Node*>(te->expr));
                auto exprMlirType = type_mapper.map_postgre_sqltype(exprTypeOid, -1, true);

                std::string scopeName;
                std::string columnName;

                if (needs_post_processing.contains(te->resno)) {
                    scopeName = finalScope;
                    columnName = "postproc_" + std::to_string(te->resno);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Adding post-processed column '%s' in scope '%s' (resno=%d)",
                            columnName.c_str(), scopeName.c_str(), te->resno);
                } else {
                    scopeName = aggrScopeName;
                    columnName = te->resname ? te->resname : "expr_" + std::to_string(te->resno);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: Adding complex expression column '%s' with type_oid=%u (resno=%d)",
                            columnName.c_str(), exprTypeOid, te->resno);
                }

                result.columns.push_back({.table_name = scopeName,
                                          .column_name = columnName,
                                          .type_oid = exprTypeOid,
                                          .typmod = -1,
                                          .mlir_type = exprMlirType,
                                          .nullable = true});
            }
        }

        for (const auto& [aggno, mapping] : aggregateMappings) {
            ctx.varno_resolution[std::make_pair(-2, aggno)] = mapping;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping to result.varno_resolution: aggno=%d -> (%s, %s)",
                    aggno, mapping.first.c_str(), mapping.second.c_str());
        }

        if (agg->plan.qual && agg->plan.qual->length > 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING clause with %d varno_resolution entries",
                    static_cast<int>(ctx.varno_resolution.size()));
            for (const auto& [key, value] : ctx.varno_resolution) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "  HAVING: varno=%d, attno=%d -> (%s, %s)", key.first, key.second,
                        value.first.c_str(), value.second.c_str());
            }

            result = apply_selection_from_qual(ctx, result, agg->plan.qual);
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning result with %zu columns, op=%p", result.columns.size(),
                static_cast<void*>(result.op));
    }
    return result;
}

} // namespace postgresql_ast
