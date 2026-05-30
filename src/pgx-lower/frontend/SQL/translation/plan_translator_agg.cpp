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
#include "utils/memutils.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "lingodb/runtime/RuntimeSpecifications.h"

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
    if (!expr) {
        return;
}
    if (IsA(expr, Aggref)) {
        result.push_back(reinterpret_cast<Aggref*>(expr));
        return;
    }

    if (IsA(expr, OpExpr)) {
        const auto* op_expr = reinterpret_cast<OpExpr*>(expr);
        ListCell* lc = nullptr;
        foreach (lc, op_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, FuncExpr)) {
        const auto* func_expr = reinterpret_cast<FuncExpr*>(expr);
        ListCell* lc = nullptr;
        foreach (lc, func_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }

    if (IsA(expr, BoolExpr)) {
        const auto* bool_expr = reinterpret_cast<BoolExpr*>(expr);
        ListCell* lc = nullptr;
        foreach (lc, bool_expr->args) {
            auto* arg = static_cast<Expr*>(lfirst(lc));
            find_all_aggrefs(arg, result);
        }
    }
}

auto get_aggregate_function(const std::string& func_name) -> mlir::relalg::AggrFunc {
    return (func_name == "sum")   ? mlir::relalg::AggrFunc::sum
           : (func_name == "avg") ? mlir::relalg::AggrFunc::avg
           : (func_name == "min") ? mlir::relalg::AggrFunc::min
           : (func_name == "max") ? mlir::relalg::AggrFunc::max
                                 : mlir::relalg::AggrFunc::count;
}

auto get_first_aggregate_argument(const Aggref* aggref) -> TargetEntry* {
    if (!aggref->args || list_length(aggref->args) == 0) {
        return nullptr;
}
    auto* arg_te = static_cast<TargetEntry*>(linitial(aggref->args));
    return (arg_te && arg_te->expr) ? arg_te : nullptr;
}

auto create_column_def(mlir::relalg::ColumnManager& column_manager, const std::string& scope_name,
                     const std::string& column_name, const mlir::Type COLUMN_TYPE) -> mlir::relalg::ColumnDefAttr {
    const auto COL_DEF = column_manager.createDef(scope_name, column_name);
    COL_DEF.getColumn().type = COLUMN_TYPE;
    return COL_DEF;
}

auto process_count_star_aggregate(mlir::OpBuilder& aggr_builder, const mlir::Location LOC, mlir::Value relation,
                               mlir::relalg::ColumnDefAttr& attr_def, mlir::relalg::ColumnManager& column_manager,
                               const std::string& aggr_scope_name, const std::string& agg_column_name,
                               std::map<int, std::pair<std::string, std::string>>& aggregate_mappings, const int AGGNO,
                               const bool LOG_DEBUG = true) -> mlir::Value {
    if (LOG_DEBUG) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Processing COUNT(*) aggregate aggno=%d", AGGNO);
    }

    auto i64_type = mlir::IntegerType::get(aggr_builder.getContext(), 64);
    attr_def = create_column_def(column_manager, aggr_scope_name, agg_column_name, i64_type);

    aggregate_mappings[AGGNO] = std::make_pair(aggr_scope_name, agg_column_name);

    if (LOG_DEBUG) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Added COUNT(*) mapping aggno=%d -> (%s, %s)", AGGNO, aggr_scope_name.c_str(),
                agg_column_name.c_str());
    }

    return aggr_builder.create<mlir::relalg::CountRowsOp>(LOC, i64_type, relation);
}

auto create_aggregate_operation(mlir::OpBuilder& aggr_builder, const mlir::Location loc, mlir::Type resultType,
                              mlir::relalg::AggrFunc aggrFuncEnum, mlir::Value relation,
                              mlir::relalg::ColumnRefAttr columnRef, const bool IS_DISTINCT) -> mlir::Value {
    if (IS_DISTINCT) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "DSA will create DISTINCT hashtable spec from types");

        auto distinct_stream = aggr_builder.create<mlir::relalg::ProjectionOp>(
            loc, mlir::relalg::TupleStreamType::get(aggr_builder.getContext()),
            mlir::relalg::SetSemanticAttr::get(aggr_builder.getContext(), mlir::relalg::SetSemantic::distinct),
            relation, aggr_builder.getArrayAttr({columnRef}));

        return aggr_builder.create<mlir::relalg::AggrFuncOp>(loc, resultType, aggrFuncEnum, distinct_stream.getResult(),
                                                             columnRef);
    }         return aggr_builder.create<mlir::relalg::AggrFuncOp>(loc, resultType, aggrFuncEnum, relation, columnRef);
   
}

} // namespace

namespace postgresql_ast {

using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLASTTranslator::Impl::translate_agg(QueryCtxT& ctx, const Agg* agg) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!agg || !agg->plan.lefttree) {
        throw std::runtime_error("invalid input");
}

    auto childResult = translate_plan_node(ctx, agg->plan.lefttree);
    if (!childResult.op || (childResult.op->getNumResults() == 0u)) {
        throw std::runtime_error("Failed to translate Agg child plan");
}

    auto child_output = childResult.op->getResult(0);
    auto& column_manager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    auto group_by_attrs = std::vector<mlir::Attribute>{};
    const auto TYPE_MAPPER = PostgreSQLTypeMapper(*ctx.builder.getContext());

    const bool IS_COMBINING = (agg->aggsplit & 0x01) != 0; // AGGSPLITOP_COMBINE flag
    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: aggsplit=%d, is_combining=%d", agg->aggsplit, IS_COMBINING);

    {
        if (agg->numCols > 0 && agg->grpColIdx) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Building GROUP BY from grpColIdx, numCols=%d", agg->numCols);
            for (int i = 0; i < agg->numCols; i++) {
                int const col_idx = agg->grpColIdx[i];
                if (col_idx > 0 && col_idx <= static_cast<int>(childResult.columns.size())) {
                    const auto& child_col = childResult.columns[col_idx - 1];
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: GROUP BY column %d: table='%s' name='%s' (from child column at index %d)", i,
                            child_col.table_name.c_str(), child_col.column_name.c_str(), col_idx - 1);
                    auto col_ref = column_manager.createRef(child_col.table_name, child_col.column_name);
                    col_ref.getColumn().type = child_col.mlir_type;
                    group_by_attrs.push_back(col_ref);
                }
            }
        }

        // Scan targetlist for additional GROUP BY columns
        // Partial aggs may pass through functionally dependent columns with ressortgroupref=0
        // RelAlg doesn't support implicit passthrough, so add them to GROUP BY explicitly
        if (agg->plan.targetlist) {
            const bool IS_PARTIAL_AGG = (agg->aggsplit & AGGSPLITOP_SERIALIZE) != 0;
            ListCell* lc = nullptr;
            foreach (lc, agg->plan.targetlist) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                // Include if: (1) explicit GROUP BY (ressortgroupref > 0), or (2) passthrough in partial agg
                const bool SHOULD_INCLUDE = ((tle != nullptr) && !tle->resjunk && IsA(tle->expr, Var))
                                            && (tle->ressortgroupref > 0 || IS_PARTIAL_AGG);
                if (SHOULD_INCLUDE) {
                    auto* var = reinterpret_cast<Var*>(tle->expr);
                    if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                        const auto& child_col = childResult.columns[var->varattno - 1];

                        // Skip if already added
                        bool already_in_group = false;
                        for (const auto& attr : group_by_attrs) {
                            if (auto existing_col_ref = mlir::dyn_cast<mlir::relalg::ColumnRefAttr>(attr)) {
                                auto existing_name = existing_col_ref.getName();
                                if (existing_name.getRootReference().str() == child_col.table_name
                                    && existing_name.getLeafReference().str() == child_col.column_name)
                                {
                                    already_in_group = true;
                                    break;
                                }
                            }
                        }

                        if (!already_in_group) {
                            auto col_ref = column_manager.createRef(child_col.table_name, child_col.column_name);
                            col_ref.getColumn().type = child_col.mlir_type;
                            group_by_attrs.push_back(col_ref);
                        }
                    }
                }
            }
        }
    }

    static size_t aggr_id = 0;
    auto aggr_scope_name = "aggr" + std::to_string(aggr_id++);
    auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    if (!agg->plan.targetlist || agg->plan.targetlist->length <= 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning childResult with %zu columns",
                childResult.columns.size());
        return childResult;
    }
    auto* block = new mlir::Block;
    block->addArgument(tuple_stream_type, ctx.builder.getUnknownLoc());
    block->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

    mlir::OpBuilder aggr_builder(ctx.builder.getContext());
    aggr_builder.setInsertionPointToStart(block);

    auto created_values = std::vector<mlir::Value>{};
    auto created_cols = std::vector<mlir::Attribute>{};

    auto aggregate_mappings = std::map<int, std::pair<std::string, std::string>>();
    auto aggregate_types = std::map<int, mlir::Type>();  // Track MLIR types for nullable info
    auto aggregate_functions = std::map<int, std::string>();  // Track function names for AVG detection

    auto needs_post_processing = std::set<int>();
    auto post_process_exprs = std::map<int, Expr*>();

    auto process_single_aggregate = [&](const Aggref* aggref, const char* resname = nullptr) -> void {
        char* const raw_func_name = get_func_name(aggref->aggfnoid);
        if (!raw_func_name) {
            PGX_WARNING("Failed to find a function name!");
            return;
        }
        const auto FUNC_NAME = std::string(raw_func_name);
        pfree(raw_func_name);

        // Track function name for spec creation
        aggregate_functions[aggref->aggno] = FUNC_NAME;

        auto agg_column_name = resname ? std::string(resname) : ("agg_" + std::to_string(aggref->aggno));
        const auto RELATION = block->getArgument(0);
        mlir::Value agg_result;

        if (FUNC_NAME == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
            // COUNT(*) - but in combining mode, we sum partial counts instead
            if (IS_COMBINING) {
                auto *const ARG_TE = get_first_aggregate_argument(aggref);
                if (!ARG_TE) {
                    PGX_ERROR("COUNT in combining mode but no argument found (aggno=%d)", aggref->aggno);
                    return;
                }

                const auto CHILD_CTX = QueryCtxT::createChildContextWithOuter(ctx, childResult);
                TranslationResult expr_context = childResult;
                expr_context.op = child_output.getDefiningOp();
                auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                    CHILD_CTX, ARG_TE->expr, expr_context, "agg_expr_" + std::to_string(aggref->aggno));

                if (stream != child_output) {
                    child_output = llvm::cast<mlir::OpResult>(stream);
                    const mlir::Type ACTUAL_TYPE = column_ref.getColumn().type;
                    const bool IS_NULLABLE = mlir::isa<mlir::db::NullableType>(ACTUAL_TYPE);

                    Oid const actual_oid = PostgreSQLTypeMapper::map_mlir_type_to_oid(ACTUAL_TYPE);
                    childResult.columns.push_back({.table_name = table_name,
                                                   .column_name = column_name,
                                                   .type_oid = actual_oid,
                                                   .typmod = exprTypmod(reinterpret_cast<Node*>(ARG_TE->expr)),
                                                   .mlir_type = ACTUAL_TYPE,
                                                   .nullable = IS_NULLABLE});
                }

                auto result_type = ctx.builder.getI64Type();
                const auto ATTR_DEF = create_column_def(column_manager, aggr_scope_name, agg_column_name, result_type);
                aggregate_mappings[aggref->aggno] = std::make_pair(aggr_scope_name, agg_column_name);
                aggregate_types[aggref->aggno] = result_type;  // Store for spec creation

                agg_result = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                    ctx.builder.getUnknownLoc(), result_type, mlir::relalg::AggrFunc::sum, RELATION, column_ref);

                PGX_LOG(AST_TRANSLATE, DEBUG, "COUNT in combining mode: using SUM on %s.%s (aggno=%d)",
                        table_name.c_str(), column_name.c_str(), aggref->aggno);

                created_cols.push_back(ATTR_DEF);
                created_values.push_back(agg_result);
            } else {
                // Normal COUNT(*) - count base rows
                mlir::relalg::ColumnDefAttr attr_def;
                agg_result = process_count_star_aggregate(aggr_builder, ctx.builder.getUnknownLoc(), RELATION, attr_def,
                                                      column_manager, aggr_scope_name, agg_column_name, aggregate_mappings,
                                                      aggref->aggno);
                created_cols.push_back(attr_def);
                created_values.push_back(agg_result);
            }
        } else {
            auto *const ARG_TE = get_first_aggregate_argument(aggref);
            if (!ARG_TE) {
                return;
}

            const auto CHILD_CTX = QueryCtxT::createChildContextWithOuter(ctx, childResult);

            TranslationResult expr_context = childResult;
            expr_context.op = child_output.getDefiningOp();
            auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                CHILD_CTX, ARG_TE->expr, expr_context, "agg_expr_" + std::to_string(aggref->aggno));

            if (stream != child_output) {
                child_output = llvm::cast<mlir::OpResult>(stream);
                const mlir::Type ACTUAL_TYPE = column_ref.getColumn().type;
                const bool IS_NULLABLE = mlir::isa<mlir::db::NullableType>(ACTUAL_TYPE);

                Oid const actual_oid = PostgreSQLTypeMapper::map_mlir_type_to_oid(ACTUAL_TYPE);
                childResult.columns.push_back({.table_name = table_name,
                                               .column_name = column_name,
                                               .type_oid = actual_oid,
                                               .typmod = exprTypmod(reinterpret_cast<Node*>(ARG_TE->expr)),
                                               .mlir_type = ACTUAL_TYPE,
                                               .nullable = IS_NULLABLE});
            }

            mlir::Type result_type;
            if (FUNC_NAME == "count") {
                result_type = ctx.builder.getI64Type();
            } else if (aggref->aggtype == BYTEAOID && aggref->aggargtypes && list_length(aggref->aggargtypes) > 0) {
                // BYTEAOID (17) indicates PostgreSQL is using polymorphic aggregate with internal state
                // Use the actual argument type for result type (works for SUM/MIN/MAX)
                Oid const arg_type_oid = lfirst_oid(list_head(aggref->aggargtypes));
                result_type = TYPE_MAPPER.map_postgre_sqltype(arg_type_oid, -1, true);
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "Polymorphic aggregate: using aggargtypes for result type: aggtype=%u -> argtype=%u",
                        aggref->aggtype, arg_type_oid);
            } else {
                result_type = TYPE_MAPPER.map_postgre_sqltype(aggref->aggtype, -1, true);
            }
            const auto ATTR_DEF = create_column_def(column_manager, aggr_scope_name, agg_column_name, result_type);
            aggregate_mappings[aggref->aggno] = std::make_pair(aggr_scope_name, agg_column_name);
            aggregate_types[aggref->aggno] = result_type;  // Store for spec creation

            // TODO: AVG in combining mode also needs special handling - it should combine using
            auto aggr_func_enum = get_aggregate_function(FUNC_NAME);
            if (FUNC_NAME == "count" && IS_COMBINING) {
                aggr_func_enum = mlir::relalg::AggrFunc::sum;
                PGX_LOG(AST_TRANSLATE, DEBUG, "COUNT with argument in combining mode: using SUM on %s.%s (aggno=%d)",
                        table_name.c_str(), column_name.c_str(), aggref->aggno);
            }
            agg_result = create_aggregate_operation(aggr_builder, ctx.builder.getUnknownLoc(), result_type, aggr_func_enum,
                                                 RELATION, column_ref, aggref->aggdistinct);
            created_cols.push_back(ATTR_DEF);
            created_values.push_back(agg_result);
        }
    };

    // Section 2: Translate expressions in the targetlist - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    {
        ListCell* lc = nullptr;
        foreach (lc, agg->plan.targetlist) {
            auto *te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr) {
                continue;
}

            if (IsA(te->expr, Aggref)) {
                auto *aggref = reinterpret_cast<Aggref*>(te->expr);
                process_single_aggregate(aggref, te->resname);
            } else {
                // This could be a complex expression with nested aggregation, like SUM(x) / SUM(y) has two aggregations
                // inside of it.
                auto nested_aggrefs = std::vector<Aggref*>();
                find_all_aggrefs(te->expr, nested_aggrefs);

                if (!nested_aggrefs.empty()) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu nested Aggrefs in complex expression at resno=%d",
                            nested_aggrefs.size(), te->resno);

                    for (auto* nested_aggref : nested_aggrefs) {
                        if (!aggregate_mappings.contains(nested_aggref->aggno)) {
                            process_single_aggregate(nested_aggref);
}
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
            ListCell* qual_lc = nullptr;
            foreach (qual_lc, agg->plan.qual) {
                auto* qual_expr = static_cast<Expr*>(lfirst(qual_lc));
                find_all_aggrefs(qual_expr, having_aggrefs);
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "Found %zu aggregate(s) in HAVING clause", having_aggrefs.size());
            for (auto* aggref : having_aggrefs) {
                if (aggregate_mappings.contains(aggref->aggno)) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Aggregate aggno=%d already in mappings, skipping", aggref->aggno);
                } else {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING aggregate aggno=%d", aggref->aggno);
                    process_single_aggregate(aggref);
                }
            }
        }
    }

    aggr_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), created_values);
    auto agg_op = ctx.builder.create<mlir::relalg::AggregationOp>(ctx.builder.getUnknownLoc(), tuple_stream_type,
                                                                 child_output, ctx.builder.getArrayAttr(group_by_attrs),
                                                                 ctx.builder.getArrayAttr(created_cols));
    agg_op.getAggrFunc().push_back(block);

    mlir::Value final_output = agg_op;
    auto final_scope = aggr_scope_name;

    // Section 4: Post processing - evaluate outer expressions that had an aggregation inside of them - - - - - - - - -
    // -
    {
        if (!needs_post_processing.empty()) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating post-processing MapOp for %zu expressions",
                    needs_post_processing.size());

            auto post_map_scope = column_manager.getUniqueScope("postmap");
            std::vector<mlir::relalg::ColumnDefAttr> post_map_cols;

            auto map_op = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), agg_op,
                                                                 ctx.builder.getArrayAttr({}));

            auto& map_region = map_op.getPredicate();
            auto* map_block = new mlir::Block;
            map_region.push_back(map_block);
            map_block->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

            auto map_builder = mlir::OpBuilder(map_block, map_block->begin());
            auto map_values = std::vector<mlir::Value>();

            auto post_proc_result = TranslationResult();
            post_proc_result.op = agg_op.getOperation();
            post_proc_result.current_scope = aggr_scope_name;
            std::map<std::pair<int, int>, std::pair<std::string, std::string>> agg_mappings;
            for (const auto& [aggno, mapping] : aggregate_mappings) {
                agg_mappings[{OUTER_VAR, aggno}] = mapping;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping for post-processing: aggno=%d -> (%s, %s)",
                        aggno, mapping.first.c_str(), mapping.second.c_str());
            }

            for (const auto RESNO : needs_post_processing) {
                auto* full_expr = post_process_exprs[RESNO];

                PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing resno=%d with expression type=%d", RESNO, full_expr->type);

                auto post_ctx = create_child_context_with_var_mappings(
                    QueryCtxT::createChildContext(ctx, map_builder, map_block->getArgument(0)), agg_mappings);
                auto post_value = translate_expression(post_ctx, full_expr);

                auto col_name = "postproc_" + std::to_string(RESNO);
                auto col_def = column_manager.createDef(post_map_scope, col_name);
                col_def.getColumn().type = post_value.getType();

                post_map_cols.push_back(col_def);
                map_values.push_back(post_value);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Created post-processing column: %s.%s", post_map_scope.c_str(),
                        col_name.c_str());
            }

            map_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), map_values);

            auto post_map_attrs = std::vector<mlir::Attribute>();
            for (const auto& col : post_map_cols) {
                post_map_attrs.push_back(col);
            }
            map_op.setComputedColsAttr(ctx.builder.getArrayAttr(post_map_attrs));

            final_output = map_op;
            final_scope = post_map_scope;

            PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing MapOp created with %zu columns", post_map_cols.size());
        }
    }

    // Section 5: Build output schema - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    TranslationResult result;
    {
        result.op = final_output.getDefiningOp();
        result.current_scope = final_scope;

        ListCell* lc = nullptr;
        foreach (lc, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr) {
                continue;
}

            if (IsA(te->expr, Aggref)) {
                auto* aggref = reinterpret_cast<Aggref*>(te->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "Second loop: Processing aggregate aggno=%d", aggref->aggno);
                auto result_type = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)
                                      ? ctx.builder.getI64Type()
                                      : TYPE_MAPPER.map_postgre_sqltype(aggref->aggtype, -1, true);

                std::string result_column_name;
                if (aggregate_mappings.contains(aggref->aggno)) {
                    const auto& mapping = aggregate_mappings[aggref->aggno];
                    result_column_name = mapping.second;
                    ctx.varno_resolution[std::make_pair(-2, aggref->aggno)] = mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Added aggregate mapping to TranslationResult: varno=-2, aggno=%d -> (%s, %s)",
                            aggref->aggno, mapping.first.c_str(), mapping.second.c_str());
                } else {
                    PGX_ERROR("Aggregate aggno=%d not found in aggregateMappings", aggref->aggno);
                    throw std::runtime_error("read logs");
                }

                PGX_LOG(AST_TRANSLATE, DEBUG, "About to push_back aggregate column: table='%s', column='%s', type_oid=%u",
                        aggr_scope_name.c_str(), result_column_name.c_str(), aggref->aggtype);
                result.columns.push_back({.table_name = aggr_scope_name,
                                          .column_name = result_column_name,
                                          .type_oid = aggref->aggtype,
                                          .typmod = -1,
                                          .mlir_type = result_type,
                                          .nullable = true});
                PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully pushed aggregate column, result now has %zu columns",
                        result.columns.size());
            } else if (IsA(te->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(te->expr);
                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    const auto& child_col = childResult.columns[var->varattno - 1];
                    bool in_group_by = false;
                    for (const auto& attr : group_by_attrs) {
                        if (auto col_ref = mlir::dyn_cast<mlir::relalg::ColumnRefAttr>(attr)) {
                            auto name = col_ref.getName();
                            if (name.getRootReference().str() == child_col.table_name
                                && name.getLeafReference().str() == child_col.column_name)
                            {
                                in_group_by = true;
                                break;
                            }
                        }
                    }

                    result.columns.push_back(child_col);
                    const char* const group_status = in_group_by ? "in GROUP BY" : "functionally dependent, pass-through";
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Adding column '%s' to output (%s, varattno=%d, resname=%s)",
                            child_col.column_name.c_str(), group_status, var->varattno,
                            te->resname ? te->resname : "<null>");
                }
            } else {
                Oid const expr_type_oid = exprType(reinterpret_cast<Node*>(te->expr));
                auto expr_mlir_type = TYPE_MAPPER.map_postgre_sqltype(expr_type_oid, -1, true);

                std::string scope_name;
                std::string column_name;

                if (needs_post_processing.contains(te->resno)) {
                    scope_name = final_scope;
                    column_name = "postproc_" + std::to_string(te->resno);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Adding post-processed column '%s' in scope '%s' (resno=%d)",
                            column_name.c_str(), scope_name.c_str(), te->resno);
                } else {
                    scope_name = aggr_scope_name;
                    column_name = te->resname ? te->resname : "expr_" + std::to_string(te->resno);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: Adding complex expression column '%s' with type_oid=%u (resno=%d)",
                            column_name.c_str(), expr_type_oid, te->resno);
                }

                result.columns.push_back({.table_name = scope_name,
                                          .column_name = column_name,
                                          .type_oid = expr_type_oid,
                                          .typmod = -1,
                                          .mlir_type = expr_mlir_type,
                                          .nullable = true});
            }
        }

        for (const auto& [aggno, mapping] : aggregate_mappings) {
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
