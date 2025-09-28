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
#include "lingodb/runtime/metadata.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

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
namespace postgresql_ast {

using namespace pgx_lower::frontend::sql::constants;

static void find_all_aggrefs(Expr* expr, std::vector<Aggref*>& result) {
    if (!expr) {
        return;
    }

    if (IsA(expr, Aggref)) {
        result.push_back(reinterpret_cast<Aggref*>(expr));
        return;
    }

    if (IsA(expr, OpExpr)) {
        auto* op_expr = reinterpret_cast<OpExpr*>(expr);
        ListCell* lc;
        foreach (lc, op_expr->args) {
            auto* arg = reinterpret_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, FuncExpr)) {
        auto* func_expr = reinterpret_cast<FuncExpr*>(expr);
        ListCell* lc;
        foreach (lc, func_expr->args) {
            auto* arg = reinterpret_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, BoolExpr)) {
        auto* bool_expr = reinterpret_cast<BoolExpr*>(expr);
        ListCell* lc;
        foreach (lc, bool_expr->args) {
            auto* arg = reinterpret_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }
}

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

    // Build GROUP BY columns from grpColIdx
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
    if (agg->plan.targetlist) {
        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (tle && tle->ressortgroupref > 0 && !tle->resjunk && IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    const auto& childCol = childResult.columns[var->varattno - 1];

                    bool alreadyInGroup = false;
                    for (const auto& attr : groupByAttrs) {
                        auto existingColRef = mlir::dyn_cast<mlir::relalg::ColumnRefAttr>(attr);
                        if (existingColRef) {
                            // Compare table and column names
                            auto existingName = existingColRef.getName();
                            if (existingName.getRootReference().str() == childCol.table_name
                                && existingName.getLeafReference().str() == childCol.column_name)
                            {
                                alreadyInGroup = true;
                                break;
                            }
                        }
                    }

                    if (!alreadyInGroup) {
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

    // Create aggregate scope
    static size_t aggrId = 0;
    std::string aggrScopeName = "aggr" + std::to_string(aggrId++);
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    if (agg->plan.targetlist && agg->plan.targetlist->length > 0) {
        auto* block = new mlir::Block;
        block->addArgument(tupleStreamType, ctx.builder.getUnknownLoc());
        block->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

        mlir::OpBuilder aggr_builder(ctx.builder.getContext());
        aggr_builder.setInsertionPointToStart(block);

        auto createdValues = std::vector<mlir::Value>{};
        auto createdCols = std::vector<mlir::Attribute>{};

        std::map<int, std::pair<std::string, std::string>> aggregateMappings;

        std::set<int> needs_post_processing;
        std::map<int, Expr*> post_process_exprs;
        std::map<int, Aggref*> post_process_aggref_map;

        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr)
                continue;

            if (IsA(te->expr, Aggref)) {
                auto aggref = reinterpret_cast<Aggref*>(te->expr);
                char* rawFuncName = get_func_name(aggref->aggfnoid);
                if (!rawFuncName)
                    continue;
                std::string funcName(rawFuncName);
                pfree(rawFuncName);

                std::string aggColumnName = te->resname ? te->resname : funcName + "_" + std::to_string(aggref->aggno);
                auto relation = block->getArgument(0);
                mlir::Value aggResult;

                if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
                    // COUNT(*)
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Creating aggregate column definition: scope='%s', column='%s' for aggno=%d",
                            aggrScopeName.c_str(), aggColumnName.c_str(), aggref->aggno);
                    auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                    attrDef.getColumn().type = ctx.builder.getI64Type();

                    aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "First loop: Added aggregate mapping aggno=%d -> (%s, %s)",
                            aggref->aggno, aggrScopeName.c_str(), aggColumnName.c_str());
                    aggResult = aggr_builder.create<mlir::relalg::CountRowsOp>(ctx.builder.getUnknownLoc(),
                                                                               ctx.builder.getI64Type(), relation);

                    // Add to created columns
                    if (attrDef && aggResult) {
                        createdCols.push_back(attrDef);
                        createdValues.push_back(aggResult);
                    }
                } else {
                    if (!aggref->args || list_length(aggref->args) == 0)
                        continue;

                    auto argTE = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (!argTE || !argTE->expr)
                        continue;

                    // For aggregate expressions, we need a special context
                    auto childCtx = QueryCtxT{ctx.current_stmt, ctx.builder, ctx.current_module, mlir::Value{},
                                              ctx.current_tuple};
                    childCtx.init_plan_results = ctx.init_plan_results;

                    auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                        childCtx, argTE->expr, childOutput, "agg_expr_" + std::to_string(aggref->aggno),
                        childResult.columns);

                    if (stream != childOutput) {
                        childOutput = cast<mlir::OpResult>(stream);

                        const auto exprOid = exprType(reinterpret_cast<Node*>(argTE->expr));
                        auto type_mapper = PostgreSQLTypeMapper(*ctx.builder.getContext());
                        auto mlirExprType = type_mapper.map_postgre_sqltype(exprOid, -1, true);

                        childResult.columns.push_back({.table_name = table_name,
                                                       .column_name = column_name,
                                                       .type_oid = exprOid,
                                                       .typmod = -1,
                                                       .mlir_type = mlirExprType,
                                                       .nullable = true});
                    }

                    auto columnRef = column_ref;

                    PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                    auto resultType = (funcName == "count") ? ctx.builder.getI64Type()
                                                            : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                    // Create the column definition now that we know we'll use it
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Creating aggregate column definition: scope='%s', column='%s' for aggno=%d",
                            aggrScopeName.c_str(), aggColumnName.c_str(), aggref->aggno);
                    auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                    attrDef.getColumn().type = resultType;

                    aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "First loop: Added aggregate mapping aggno=%d -> (%s, %s)",
                            aggref->aggno, aggrScopeName.c_str(), aggColumnName.c_str());

                    auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                        : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                        : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                        : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                              : mlir::relalg::AggrFunc::count;

                    // I thought this would be a child node, but turns out its a flag/list... neat!
                    if (aggref->aggdistinct) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Processing %s(DISTINCT) aggregate", funcName.c_str());

                        auto distinctStream = aggr_builder.create<mlir::relalg::ProjectionOp>(
                            ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(ctx.builder.getContext()),
                            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(),
                                                               mlir::relalg::SetSemantic::distinct),
                            relation, ctx.builder.getArrayAttr({columnRef}));

                        aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                            ctx.builder.getUnknownLoc(), resultType, aggrFuncEnum, distinctStream.getResult(), columnRef);
                    } else {
                        aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                            ctx.builder.getUnknownLoc(), resultType, aggrFuncEnum, relation, columnRef);
                    }

                    // Add the created aggregate to the output
                    if (attrDef && aggResult) {
                        createdCols.push_back(attrDef);
                        createdValues.push_back(aggResult);
                    }
                }
            } else if (IsA(te->expr, Var)) {
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "First loop: Skipping T_Var at resno=%d (GROUP BY column, handled in second loop)", te->resno);
            } else {
                std::vector<Aggref*> nested_aggrefs;
                find_all_aggrefs(te->expr, nested_aggrefs);

                if (!nested_aggrefs.empty()) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu nested Aggrefs in complex expression at resno=%d",
                            nested_aggrefs.size(), te->resno);

                    Aggref* first_aggref = nested_aggrefs[0];
                    for (Aggref* nested_aggref : nested_aggrefs) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Processing nested Aggref aggno=%d in expression at resno=%d",
                                nested_aggref->aggno, te->resno);

                        char* rawFuncName = get_func_name(nested_aggref->aggfnoid);
                        if (!rawFuncName) {
                            PGX_WARNING("Could not get function name for nested Aggref aggno=%d", nested_aggref->aggno);
                            continue;
                        }
                        std::string funcName(rawFuncName);
                        pfree(rawFuncName);

                        std::string aggColumnName = "nested_agg_" + std::to_string(nested_aggref->aggno);
                        auto relation = block->getArgument(0);
                        mlir::Value aggResult;

                        if (funcName == "count" && (!nested_aggref->args || list_length(nested_aggref->args) == 0)) {
                            auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                            attrDef.getColumn().type = ctx.builder.getI64Type();

                            aggregateMappings[nested_aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                            PGX_LOG(AST_TRANSLATE, DEBUG,
                                    "First loop (nested): Added aggregate mapping aggno=%d -> (%s, %s)",
                                    nested_aggref->aggno, aggrScopeName.c_str(), aggColumnName.c_str());

                            aggResult = aggr_builder.create<mlir::relalg::CountRowsOp>(
                                ctx.builder.getUnknownLoc(), ctx.builder.getI64Type(), relation);

                            if (attrDef && aggResult) {
                                createdCols.push_back(attrDef);
                                createdValues.push_back(aggResult);
                            }
                        } else {
                            if (!nested_aggref->args || list_length(nested_aggref->args) == 0) {
                                PGX_WARNING("Nested Aggref aggno=%d has no arguments", nested_aggref->aggno);
                                continue;
                            }

                            auto argTE = static_cast<TargetEntry*>(linitial(nested_aggref->args));
                            if (!argTE || !argTE->expr) {
                                PGX_WARNING("Nested Aggref aggno=%d has invalid argument", nested_aggref->aggno);
                                continue;
                            }

                            auto childCtx = QueryCtxT{ctx.current_stmt, ctx.builder, ctx.current_module, mlir::Value{},
                                                      ctx.current_tuple};
                            childCtx.init_plan_results = ctx.init_plan_results;

                            auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                                childCtx, argTE->expr, childOutput,
                                "nested_agg_expr_" + std::to_string(nested_aggref->aggno), childResult.columns);

                            if (stream != childOutput) {
                                childOutput = cast<mlir::OpResult>(stream);

                                const auto exprOid = exprType(reinterpret_cast<Node*>(argTE->expr));
                                auto type_mapper = PostgreSQLTypeMapper(*ctx.builder.getContext());
                                auto mlirExprType = type_mapper.map_postgre_sqltype(exprOid, -1, true);

                                childResult.columns.push_back({.table_name = table_name,
                                                               .column_name = column_name,
                                                               .type_oid = exprOid,
                                                               .typmod = -1,
                                                               .mlir_type = mlirExprType,
                                                               .nullable = true});
                            }

                            auto columnRef = column_ref;

                            PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                            auto resultType = (funcName == "count")
                                                  ? ctx.builder.getI64Type()
                                                  : type_mapper.map_postgre_sqltype(nested_aggref->aggtype, -1, true);

                            auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                            attrDef.getColumn().type = resultType;

                            aggregateMappings[nested_aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                            PGX_LOG(AST_TRANSLATE, DEBUG,
                                    "First loop (nested): Added aggregate mapping aggno=%d -> (%s, %s)",
                                    nested_aggref->aggno, aggrScopeName.c_str(), aggColumnName.c_str());

                            auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                                : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                                : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                                : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                                      : mlir::relalg::AggrFunc::count;

                            if (nested_aggref->aggdistinct) {
                                PGX_LOG(AST_TRANSLATE, DEBUG, "Processing nested %s(DISTINCT) aggregate",
                                        funcName.c_str());

                                auto distinctStream = aggr_builder.create<mlir::relalg::ProjectionOp>(
                                    ctx.builder.getUnknownLoc(),
                                    mlir::relalg::TupleStreamType::get(ctx.builder.getContext()),
                                    mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(),
                                                                       mlir::relalg::SetSemantic::distinct),
                                    relation, ctx.builder.getArrayAttr({columnRef}));

                                aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                                    ctx.builder.getUnknownLoc(), resultType, aggrFuncEnum, distinctStream.getResult(),
                                    columnRef);
                            } else {
                                aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                                    ctx.builder.getUnknownLoc(), resultType, aggrFuncEnum, relation, columnRef);
                            }

                            if (attrDef && aggResult) {
                                createdCols.push_back(attrDef);
                                createdValues.push_back(aggResult);
                            }
                        }
                    }

                    needs_post_processing.insert(te->resno);
                    post_process_exprs[te->resno] = te->expr;
                    post_process_aggref_map[te->resno] = first_aggref;

                    PGX_LOG(AST_TRANSLATE, DEBUG, "Marked resno=%d for post-processing (full expr with %zu aggregates)",
                            te->resno, nested_aggrefs.size());
                } else {
                    PGX_WARNING("Unexpected non-aggregate, non-Var expression in aggregate targetlist at resno=%d, "
                                "type=%d",
                                te->resno, te->expr->type);
                }
            }
        }

        // Process aggregates from HAVING clause that aren't in targetlist
        if (agg->plan.qual && agg->plan.qual->length > 0) {
            std::vector<Aggref*> having_aggrefs;
            ListCell* qual_lc;
            foreach (qual_lc, agg->plan.qual) {
                auto* qual_expr = reinterpret_cast<Expr*>(lfirst(qual_lc));
                find_all_aggrefs(qual_expr, having_aggrefs);
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu aggregate(s) in HAVING clause", having_aggrefs.size());

            for (auto* aggref : having_aggrefs) {
                // Skip if already processed
                if (aggregateMappings.find(aggref->aggno) != aggregateMappings.end()) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Aggregate aggno=%d already in mappings, skipping", aggref->aggno);
                    continue;
                }

                char* rawFuncName = get_func_name(aggref->aggfnoid);
                if (!rawFuncName) {
                    PGX_WARNING("Could not get function name for HAVING aggregate aggno=%d", aggref->aggno);
                    continue;
                }
                std::string funcName(rawFuncName);
                pfree(rawFuncName);

                std::string aggColumnName = funcName + "_" + std::to_string(aggref->aggno);
                auto relation = block->getArgument(0);
                mlir::Value aggResult;

                PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING aggregate: func=%s, aggno=%d, column=%s",
                        funcName.c_str(), aggref->aggno, aggColumnName.c_str());

                if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
                    // COUNT(*)
                    auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                    attrDef.getColumn().type = ctx.builder.getI64Type();

                    aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "HAVING: Added COUNT(*) mapping aggno=%d -> (%s, %s)", aggref->aggno,
                            aggrScopeName.c_str(), aggColumnName.c_str());

                    aggResult = aggr_builder.create<mlir::relalg::CountRowsOp>(ctx.builder.getUnknownLoc(),
                                                                               ctx.builder.getI64Type(), relation);

                    createdCols.push_back(attrDef);
                    createdValues.push_back(aggResult);
                } else {
                    // SUM, AVG, MIN, MAX, etc.
                    if (!aggref->args || list_length(aggref->args) == 0) {
                        PGX_WARNING("HAVING aggregate aggno=%d has no arguments", aggref->aggno);
                        continue;
                    }

                    auto argTE = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (!argTE || !argTE->expr) {
                        PGX_WARNING("HAVING aggregate aggno=%d has invalid argument", aggref->aggno);
                        continue;
                    }

                    // Translate aggregate argument expression
                    auto childCtx = QueryCtxT{ctx.current_stmt, ctx.builder, ctx.current_module, mlir::Value{},
                                              ctx.current_tuple};
                    childCtx.init_plan_results = ctx.init_plan_results;

                    auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                        childCtx, argTE->expr, childOutput, "having_agg_expr_" + std::to_string(aggref->aggno),
                        childResult.columns);

                    if (stream != childOutput) {
                        childOutput = cast<mlir::OpResult>(stream);

                        const auto exprOid = exprType(reinterpret_cast<Node*>(argTE->expr));
                        auto type_mapper = PostgreSQLTypeMapper(*ctx.builder.getContext());
                        auto mlirExprType = type_mapper.map_postgre_sqltype(exprOid, -1, true);

                        childResult.columns.push_back({.table_name = table_name,
                                                       .column_name = column_name,
                                                       .type_oid = exprOid,
                                                       .typmod = -1,
                                                       .mlir_type = mlirExprType,
                                                       .nullable = true});
                    }

                    PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                    auto resultType = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)
                                          ? ctx.builder.getI64Type()
                                          : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                    auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                    attrDef.getColumn().type = resultType;

                    aggregateMappings[aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "HAVING: Added aggregate mapping aggno=%d -> (%s, %s)", aggref->aggno,
                            aggrScopeName.c_str(), aggColumnName.c_str());

                    auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                        : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                        : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                        : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                              : mlir::relalg::AggrFunc::count;

                    aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(ctx.builder.getUnknownLoc(), resultType,
                                                                              aggrFuncEnum, relation, column_ref);

                    createdCols.push_back(attrDef);
                    createdValues.push_back(aggResult);
                }
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "After processing HAVING aggregates: aggregateMappings has %zu entries",
                    aggregateMappings.size());
        }

        aggr_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), createdValues);

        auto aggOp = ctx.builder.create<mlir::relalg::AggregationOp>(ctx.builder.getUnknownLoc(), tupleStreamType,
                                                                     childOutput, ctx.builder.getArrayAttr(groupByAttrs),
                                                                     ctx.builder.getArrayAttr(createdCols));
        aggOp.getAggrFunc().push_back(block);

        mlir::Value finalOutput = aggOp;
        std::string finalScope = aggrScopeName;

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

            mlir::OpBuilder mapBuilder(mapBlock, mapBlock->begin());
            std::vector<mlir::Value> mapValues;

            TranslationResult postProcResult;
            postProcResult.op = aggOp.getOperation();
            postProcResult.current_scope = aggrScopeName;
            for (const auto& [aggno, mapping] : aggregateMappings) {
                postProcResult.varno_resolution[std::make_pair(-2, aggno)] = mapping;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping for post-processing: aggno=%d -> (%s, %s)",
                        aggno, mapping.first.c_str(), mapping.second.c_str());
            }

            for (int resno : needs_post_processing) {
                Expr* full_expr = post_process_exprs[resno];
                Aggref* nested_aggref = post_process_aggref_map[resno];

                PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing resno=%d, aggno=%d", resno, nested_aggref->aggno);

                auto& [aggr_scope, aggr_colname] = aggregateMappings.at(nested_aggref->aggno);
                auto aggr_colref = columnManager.createRef(aggr_scope, aggr_colname);

                mlir::Value aggr_value = mapBuilder
                                             .create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(),
                                                                                aggr_colref.getColumn().type,
                                                                                aggr_colref, mapBlock->getArgument(0))
                                             .getRes();

                auto postCtx = QueryCtxT{ctx.current_stmt, mapBuilder, ctx.current_module, mapBlock->getArgument(0),
                                         ctx.outer_tuple};
                postCtx.init_plan_results = ctx.init_plan_results;

                mlir::Value post_value = aggr_value;

                if (IsA(full_expr, OpExpr)) {
                    auto* op_expr = reinterpret_cast<OpExpr*>(full_expr);

                    ListCell* arg_lc;
                    std::vector<mlir::Value> arg_values;
                    foreach (arg_lc, op_expr->args) {
                        auto* arg_expr = reinterpret_cast<Expr*>(lfirst(arg_lc));

                        if (IsA(arg_expr, Aggref)) {
                            auto* arg_aggref = reinterpret_cast<Aggref*>(arg_expr);
                            auto& [arg_scope, arg_colname] = aggregateMappings.at(arg_aggref->aggno);
                            auto arg_colref = columnManager.createRef(arg_scope, arg_colname);
                            mlir::Value arg_aggr_value = mapBuilder
                                                             .create<mlir::relalg::GetColumnOp>(
                                                                 ctx.builder.getUnknownLoc(), arg_colref.getColumn().type,
                                                                 arg_colref, mapBlock->getArgument(0))
                                                             .getRes();
                            arg_values.push_back(arg_aggr_value);
                        } else {
                            auto arg_val = translate_expression(postCtx, arg_expr, postProcResult);
                            arg_values.push_back(arg_val);
                        }
                    }

                    if (arg_values.size() == 2) {
                        char* op_name = get_opname(op_expr->opno);
                        auto loc = ctx.builder.getUnknownLoc();
                        mlir::SmallVector<mlir::Type, 1> inferredTypes;

                        if (op_name && strcmp(op_name, "*") == 0) {
                            if (mlir::failed(mlir::db::MulOp::inferReturnTypes(ctx.builder.getContext(), loc,
                                                                               {arg_values[0], arg_values[1]}, nullptr,
                                                                               nullptr, {}, inferredTypes)))
                            {
                                PGX_ERROR("Failed to infer MulOp return type in post-processing");
                                throw std::runtime_error("Check logs");
                            }
                            post_value = mapBuilder.create<mlir::db::MulOp>(loc, inferredTypes[0], arg_values[0],
                                                                            arg_values[1]);
                        } else if (op_name && strcmp(op_name, "/") == 0) {
                            if (mlir::failed(mlir::db::DivOp::inferReturnTypes(ctx.builder.getContext(), loc,
                                                                               {arg_values[0], arg_values[1]}, nullptr,
                                                                               nullptr, {}, inferredTypes)))
                            {
                                PGX_ERROR("Failed to infer DivOp return type in post-processing");
                                throw std::runtime_error("Check logs");
                            }
                            post_value = mapBuilder.create<mlir::db::DivOp>(loc, inferredTypes[0], arg_values[0],
                                                                            arg_values[1]);
                        } else if (op_name && strcmp(op_name, "+") == 0) {
                            if (mlir::failed(mlir::db::AddOp::inferReturnTypes(ctx.builder.getContext(), loc,
                                                                               {arg_values[0], arg_values[1]}, nullptr,
                                                                               nullptr, {}, inferredTypes)))
                            {
                                PGX_ERROR("Failed to infer AddOp return type in post-processing");
                                throw std::runtime_error("Check logs");
                            }
                            post_value = mapBuilder.create<mlir::db::AddOp>(loc, inferredTypes[0], arg_values[0],
                                                                            arg_values[1]);
                        } else if (op_name && strcmp(op_name, "-") == 0) {
                            if (mlir::failed(mlir::db::SubOp::inferReturnTypes(ctx.builder.getContext(), loc,
                                                                               {arg_values[0], arg_values[1]}, nullptr,
                                                                               nullptr, {}, inferredTypes)))
                            {
                                PGX_ERROR("Failed to infer SubOp return type in post-processing");
                                throw std::runtime_error("Check logs");
                            }
                            post_value = mapBuilder.create<mlir::db::SubOp>(loc, inferredTypes[0], arg_values[0],
                                                                            arg_values[1]);
                        }

                        if (op_name)
                            pfree(op_name);
                    }
                }

                std::string colName = "postproc_" + std::to_string(resno);
                auto colDef = columnManager.createDef(postMapScope, colName);
                colDef.getColumn().type = post_value.getType();

                postMapCols.push_back(colDef);
                mapValues.push_back(post_value);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Created post-processing column: %s.%s", postMapScope.c_str(),
                        colName.c_str());
            }

            mapBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mapValues);

            std::vector<mlir::Attribute> postMapAttrs;
            for (const auto& col : postMapCols) {
                postMapAttrs.push_back(col);
            }
            mapOp.setComputedColsAttr(ctx.builder.getArrayAttr(postMapAttrs));

            finalOutput = mapOp;
            finalScope = postMapScope;

            PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing MapOp created with %zu columns", postMapCols.size());
        }

        // Build output schema
        TranslationResult result;
        result.op = finalOutput.getDefiningOp();
        result.current_scope = finalScope;

        foreach (lc, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr)
                continue;

            if (IsA(te->expr, Aggref)) {
                auto* aggref = reinterpret_cast<Aggref*>(te->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "Second loop: Processing aggregate aggno=%d", aggref->aggno);
                PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                auto resultType = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)
                                      ? ctx.builder.getI64Type()
                                      : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                // Use the same column name that was used in aggregateMappings
                std::string resultColumnName;
                if (aggregateMappings.find(aggref->aggno) != aggregateMappings.end()) {
                    const auto& mapping = aggregateMappings[aggref->aggno];
                    resultColumnName = mapping.second; // Use the same column name from the mapping
                    result.varno_resolution[std::make_pair(-2, aggref->aggno)] = mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Added aggregate mapping to TranslationResult: varno=-2, aggno=%d -> (%s, %s)",
                            aggref->aggno, mapping.first.c_str(), mapping.second.c_str());
                } else {
                    PGX_WARNING("Aggregate aggno=%d not found in aggregateMappings", aggref->aggno);
                    // Fallback if not in mappings (shouldn't happen)
                    resultColumnName = te->resname ? te->resname : "agg_" + std::to_string(te->resno);
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
                PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                auto exprMlirType = type_mapper.map_postgre_sqltype(exprTypeOid, -1, true);

                std::string scopeName;
                std::string columnName;

                if (needs_post_processing.count(te->resno) > 0) {
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

        // Add all aggregate mappings to varno_resolution for HAVING clause processing
        for (const auto& [aggno, mapping] : aggregateMappings) {
            result.varno_resolution[std::make_pair(-2, aggno)] = mapping;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping to result.varno_resolution: aggno=%d -> (%s, %s)",
                    aggno, mapping.first.c_str(), mapping.second.c_str());
        }

        if (agg->plan.qual && agg->plan.qual->length > 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING clause with %d varno_resolution entries",
                    static_cast<int>(result.varno_resolution.size()));
            for (const auto& [key, value] : result.varno_resolution) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "  HAVING: varno=%d, attno=%d -> (%s, %s)", key.first, key.second,
                        value.first.c_str(), value.second.c_str());
            }

            result = apply_selection_from_qual(ctx, result, agg->plan.qual);
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning result with %zu columns, op=%p", result.columns.size(),
                static_cast<void*>(result.op));
        return result;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning childResult with %zu columns", childResult.columns.size());
    return childResult;
}

} // namespace postgresql_ast
