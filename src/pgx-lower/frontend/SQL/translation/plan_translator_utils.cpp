#include "translator_internals.h"
#include "lingodb/runtime/PgSortRuntime.h"
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
#include "catalog/pg_operator.h"
#include "catalog/namespace.h"
#include "fmgr.h"

extern "C" Oid compatible_oper_opid(List* op, Oid arg1, Oid arg2, bool no_error);
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
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"

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

auto PostgreSQLASTTranslator::Impl::translate_plan_node(QueryCtxT& ctx, Plan* plan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!plan) {
        PGX_ERROR("Plan node is null");
        throw std::runtime_error("Plan node is null");
    }

    const size_t PARAMS_BEFORE = ctx.params.size();
    process_init_plans(ctx, plan);
    PGX_LOG(AST_TRANSLATE, DEBUG, "After processing InitPlans: context has %zu InitPlans (%zu new)", ctx.params.size(),
            ctx.params.size() - PARAMS_BEFORE);

    TranslationResult result;

    switch (plan->type) {
    case T_IndexScan: result = translate_index_scan(ctx, reinterpret_cast<IndexScan*>(plan)); break;
    case T_IndexOnlyScan: result = translate_index_only_scan(ctx, reinterpret_cast<IndexOnlyScan*>(plan)); break;
    case T_SeqScan: result = translate_seq_scan(ctx, reinterpret_cast<SeqScan*>(plan)); break;
    case T_BitmapHeapScan: result = translate_bitmap_heap_scan(ctx, reinterpret_cast<BitmapHeapScan*>(plan)); break;
    case T_Agg: result = translate_agg(ctx, reinterpret_cast<Agg*>(plan)); break;
    case T_Sort: result = translate_sort(ctx, reinterpret_cast<Sort*>(plan)); break;
    case T_IncrementalSort: result = translate_sort(ctx, reinterpret_cast<Sort*>(plan)); break;
    case T_Limit: result = translate_limit(ctx, reinterpret_cast<Limit*>(plan)); break;
    case T_Gather: result = translate_gather(ctx, reinterpret_cast<Gather*>(plan)); break;
    case T_GatherMerge: result = translate_gather_merge(ctx, reinterpret_cast<GatherMerge*>(plan)); break;
    case T_MergeJoin: result = translate_merge_join(ctx, reinterpret_cast<MergeJoin*>(plan)); break;
    case T_HashJoin: result = translate_hash_join(ctx, reinterpret_cast<HashJoin*>(plan)); break;
    case T_Hash: result = translate_hash(ctx, reinterpret_cast<Hash*>(plan)); break;
    case T_NestLoop: result = translate_nest_loop(ctx, reinterpret_cast<NestLoop*>(plan)); break;
    case T_Material: result = translate_material(ctx, reinterpret_cast<Material*>(plan)); break;
    case T_Memoize: result = translate_memoize(ctx, reinterpret_cast<Memoize*>(plan)); break;
    case T_SubqueryScan: result = translate_subquery_scan(ctx, reinterpret_cast<SubqueryScan*>(plan)); break;
    case T_CteScan: result = translate_cte_scan(ctx, reinterpret_cast<CteScan*>(plan)); break;
    default: PGX_ERROR("Unsupported plan node type: %d", plan->type); result.op = nullptr;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_plan_node returning result with %zu columns", result.columns.size());
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_sort(QueryCtxT& ctx, const Sort* sort) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!sort || !sort->plan.lefttree) {
        PGX_ERROR("Invalid Sort parameters or missing child");
        return TranslationResult{};
    }

    auto child_result = translate_plan_node(ctx, sort->plan.lefttree);
    if (!child_result.op) {
        PGX_ERROR("Failed to translate Sort child plan");
        return child_result;
    }
    PGX_LOG(AST_TRANSLATE, DEBUG, "Sort node got %s", child_result.toString().data());

    if ((sort->numCols == 0) || !sort->sortColIdx) {
        return child_result;
    }

    auto& column_manager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    std::vector<mlir::Attribute> sort_specs;
    for (int i = 0; i < sort->numCols; i++) {
        const AttrNumber COL_IDX = sort->sortColIdx[i];
        if (COL_IDX <= 0 || COL_IDX >= MAX_COLUMN_INDEX) {
            continue;
}

        auto spec = mlir::relalg::SortSpec::asc;
        if (sort->sortOperators) {
            if (char* const oprname = get_opname(sort->sortOperators[i])) {
                spec = (std::string(oprname) == ">" || std::string(oprname) == ">=") ? mlir::relalg::SortSpec::desc
                                                                                     : mlir::relalg::SortSpec::asc;
                pfree(oprname);
            }
        }

        ListCell* lc = nullptr;
        int idx = 0;
        foreach (lc, sort->plan.targetlist) {
            if (++idx != COL_IDX) {
                continue;
}

            const TargetEntry* const tle = static_cast<TargetEntry*>(lfirst(lc));
            if (IsA(tle->expr, Var)) {
                const Var* const var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= child_result.columns.size()) {
                    const auto& column = child_result.columns[var->varattno - 1];
                    sort_specs.push_back(mlir::relalg::SortSpecificationAttr::get(
                        ctx.builder.getContext(), column_manager.createRef(column.table_name, column.column_name), spec));
                }
            }
            break;
        }
    }

    if (sort_specs.empty()) {
        return child_result;
    }

    auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    const auto SORT_OP = ctx.builder.create<mlir::relalg::SortOp>(
        ctx.builder.getUnknownLoc(), tuple_stream_type, child_result.op->getResult(0), ctx.builder.getArrayAttr(sort_specs));

    TranslationResult result;
    result.op = SORT_OP;

    if (sort->plan.targetlist) {
        result.columns.clear();
        ListCell* lc = nullptr;
        foreach (lc, sort->plan.targetlist) {
            const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (!tle) {
                continue;
}

            if (tle->expr && IsA(tle->expr, Var)) {
                const auto* var = reinterpret_cast<Var*>(tle->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "Sort targetentry: resjunk=%d, varattno=%d, childResult.columns.size()=%zu",
                        tle->resjunk, var->varattno, child_result.columns.size());
                if (var->varattno > 0 && var->varattno <= child_result.columns.size()) {
                    const auto& col = child_result.columns[var->varattno - 1];
                    PGX_LOG(AST_TRANSLATE, DEBUG, "  Adding column: %s.%s", col.table_name.c_str(), col.column_name.c_str());
                    result.columns.push_back(col);
                }
            }
        }
    } else {
        result.columns = child_result.columns;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Sort returning %zu columns:", result.columns.size());
    for (size_t i = 0; i < result.columns.size(); i++) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s", i, result.columns[i].table_name.c_str(), result.columns[i].column_name.c_str());
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_limit(QueryCtxT& ctx, const Limit* limit) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!limit) {
        PGX_ERROR("Invalid Limit parameters");
        return TranslationResult{};
    }

    TranslationResult child_result;

    if (Plan* const left_tree = limit->plan.lefttree) {
        child_result = translate_plan_node(ctx, left_tree);
        if (!child_result.op) {
            PGX_ERROR("Failed to translate Limit child plan");
            return child_result;
        }
    } else {
        PGX_WARNING("Limit node has no child plan");
        return TranslationResult{};
    }

    auto child_output = child_result.op->getResult(0);
    if (!child_output) {
        PGX_ERROR("Child operation has no result");
        return child_result;
    }

    int64_t limit_count = DEFAULT_LIMIT_COUNT;
    int64_t limit_offset = 0;

    Node* const limit_offset_node = limit->limitOffset;

    if (Node* const limit_count_node = limit->limitCount) {
        Node* const node = limit_count_node;
        if (IsA(node, Const)) {
            const Const* const const_node = reinterpret_cast<Const*>(node);
            if (!const_node->constisnull) {
                limit_count = static_cast<int64_t>(const_node->constvalue);
            }
        } else {
            PGX_WARNING("Limit count is not a Const or Param node");
        }
    }

    if (limit_offset_node) {
        Node* const node = limit_offset_node;
        if (IsA(node, Const)) {
            const Const* const const_node = reinterpret_cast<Const*>(node);
            if (!const_node->constisnull) {
                limit_offset = static_cast<int64_t>(const_node->constvalue);
            }
        }
    }

    if (limit_count < 0) {
        PGX_WARNING("Invalid negative limit count: %d", limit_count);
        limit_count = DEFAULT_LIMIT_COUNT;
    } else if (limit_count > MAX_LIMIT_COUNT) {
        PGX_WARNING("Very large limit count: %d", limit_count);
    }

    if (limit_offset < 0) {
        PGX_WARNING("Negative offset not supported, using 0");
        limit_offset = 0;
    }

    if (limit_count == -1) {
        limit_count = INT32_MAX;
    }

    const auto LIMIT_OP = ctx.builder.create<mlir::relalg::LimitOp>(
        ctx.builder.getUnknownLoc(), ctx.builder.getI32IntegerAttr(static_cast<int32_t>(limit_count)), child_output);

    TranslationResult result;
    result.op = LIMIT_OP;
    result.columns = child_result.columns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_gather(QueryCtxT& ctx, const Gather* gather) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!gather) {
        PGX_ERROR("Invalid Gather parameters");
        return TranslationResult{};
    }

    TranslationResult child_result;

    if (Plan* const left_tree = gather->plan.lefttree) {
        child_result = translate_plan_node(ctx, left_tree);
        if (!child_result.op) {
            PGX_ERROR("Failed to translate Gather child plan");
            return child_result;
        }
    } else {
        PGX_WARNING("Gather node has no child plan");
        return TranslationResult{};
    }

    return child_result;
}

auto PostgreSQLASTTranslator::Impl::translate_gather_merge(QueryCtxT& ctx, const GatherMerge* gather_merge)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!gather_merge) {
        PGX_ERROR("Invalid GatherMerge parameters");
        return TranslationResult{};
    }

    TranslationResult child_result;
    if (Plan* const left_tree = gather_merge->plan.lefttree) {
        child_result = translate_plan_node(ctx, left_tree);
        if (!child_result.op) {
            PGX_ERROR("Failed to translate GatherMerge child plan");
            return child_result;
        }
    } else {
        PGX_WARNING("GatherMerge node has no child plan");
        return TranslationResult{};
    }

    // GatherMerge is parallel execution coordinator - pass through child for now
    // Note: Ignoring sort columns (numCols, sortColIdx, etc.) as child already sorted
    PGX_LOG(AST_TRANSLATE, DEBUG, "GatherMerge: passing through child result (parallel gathering not implemented)");
    return child_result;
}

auto PostgreSQLASTTranslator::Impl::translate_material(QueryCtxT& ctx, const Material* material) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!material || !material->plan.lefttree) {
        PGX_ERROR("Invalid Material parameters");
        throw std::runtime_error("Invalid Material parameters");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Material node is a pass-through, translating its child");
    return translate_plan_node(ctx, material->plan.lefttree);
}

auto PostgreSQLASTTranslator::Impl::translate_memoize(QueryCtxT& ctx, const Memoize* memoize) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!memoize || !memoize->plan.lefttree) {
        PGX_ERROR("Invalid Memoize parameters");
        throw std::runtime_error("Invalid Memoize parameters");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Memoize node is a pass-through (caching handled by JIT), translating its child");
    return translate_plan_node(ctx, memoize->plan.lefttree);
}

auto PostgreSQLASTTranslator::Impl::process_init_plans(QueryCtxT& ctx, const Plan* plan) -> void {
    PGX_IO(AST_TRANSLATE);

    if (!plan->initPlan || list_length(plan->initPlan) == 0) {
        return;
    }

    const List* const all_subplans = ctx.current_stmt.subplans;
    const int NUM_SUBPLANS = list_length(all_subplans);
    ListCell* lc = nullptr;
    foreach (lc, plan->initPlan) {
        const auto* subplan = static_cast<SubPlan*>(lfirst(lc));
        if (!subplan) {
            PGX_ERROR("Invalid SubPlan in initPlan list");
            continue;
        }

        const int PLAN_ID = subplan->plan_id;
        if (PLAN_ID < 1 || PLAN_ID > NUM_SUBPLANS) {
            PGX_ERROR("SubPlan plan_id %d out of range (have %d subplans)", PLAN_ID, NUM_SUBPLANS);
            continue;
        }

        auto* initplan = static_cast<Plan*>(list_nth(all_subplans, PLAN_ID - 1));
        if (!initplan) {
            PGX_ERROR("SubPlan plan_id %d points to null Plan", PLAN_ID);
            continue;
        }

        auto initplan_result = translate_plan_node(ctx, initplan);
        if (!initplan_result.op) {
            PGX_ERROR("Failed to translate InitPlan (plan_id=%d)", PLAN_ID);
            continue;
        }

        const List* const set_param = subplan->setParam;
        if (!set_param || list_length(set_param) == 0) {
            PGX_ERROR("InitPlan has no setParam");
            continue;
        }
        const int PARAMID = list_nth_int(set_param, 0);

        if (initplan_result.columns.empty()) {
            PGX_ERROR("InitPlan produced no columns for paramid=%d", PARAMID);
            continue;
        }

        const bool IS_CTE = (subplan->subLinkType == 7) || (initplan_result.columns.size() > 1);
        if (IS_CTE) {
            ctx.initplan_results[PARAMID] = initplan_result;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Stored CTE InitPlan result for paramid=%d (plan_id=%d, %zu columns)",
                    PARAMID, PLAN_ID, initplan_result.columns.size());
        } else {
            const auto& col = initplan_result.columns[0];
            ctx.params[PARAMID] = pgx_lower::frontend::sql::ResolvedParam{
                .table_name = col.table_name,
                .column_name = col.column_name,
                .type_oid = col.type_oid,
                .typmod = col.typmod,
                .nullable = col.nullable,
                .mlir_type = col.mlir_type,
                .cached_value = initplan_result.op->getResult(0)};
            PGX_LOG(AST_TRANSLATE, DEBUG, "Stored scalar InitPlan result for paramid=%d (plan_id=%d)", PARAMID, PLAN_ID);
        }
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Processed %d InitPlans, context now has %zu total", list_length(plan->initPlan),
            ctx.params.size());
}

auto PostgreSQLASTTranslator::Impl::create_query_function(mlir::OpBuilder& builder) -> mlir::func::FuncOp {
    PGX_IO(AST_TRANSLATE);
    auto table_type = mlir::dsa::TableType::get(builder.getContext());
    auto query_func_type = builder.getFunctionType({}, {table_type});
    auto query_func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), QUERY_FUNCTION_NAME, query_func_type);

    auto& query_body = query_func.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&query_body);

    return query_func;
}

auto PostgreSQLASTTranslator::Impl::apply_selection_from_qual(const QueryCtxT& ctx, const TranslationResult& input,
                                                              const List* qual) -> TranslationResult {
    // This applies a filter to the SELECT statement, so it can be a SELECT x FROM y WHERE z, or a
    // GROUP BY HAVING statement for instance.
    PGX_IO(AST_TRANSLATE);
    if (!input.op || !qual || qual->length == 0) {
        return input;
    }

    auto input_value = input.op->getResult(0);
    if (!input_value) {
        PGX_ERROR("Input operation has no result");
        throw std::runtime_error("Input operation has no result");
    }

    auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), input_value);

    { // Build the predicate region
        auto& predicate_region = selection_op.getPredicate();
        auto* predicate_block = new mlir::Block;
        predicate_region.push_back(predicate_block);

        const auto TUPLE_TYPE = mlir::relalg::TupleType::get(&context_);
        const auto TUPLE_ARG = predicate_block->addArgument(TUPLE_TYPE, ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicate_block);

        const auto TMP_CTX = QueryCtxT::createChildContext(ctx, predicate_builder, TUPLE_ARG);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu params", TMP_CTX.params.size());

        mlir::Value predicate_result = nullptr;
        if (qual && qual->length > 0) {
            if (!qual->elements) {
                PGX_WARNING("Qual list has length but no elements array - continuing without filter");
            } else {
                for (int i = 0; i < qual->length; i++) {
                    const ListCell* const lc = &qual->elements[i];
                    if (!lc) {
                        PGX_WARNING("Null ListCell at index %d", i);
                        continue;
                    }

                    auto *const QUAL_NODE = static_cast<Node*>(lfirst(lc));

                    if (!QUAL_NODE) {
                        PGX_WARNING("Null qual node at index %d", i);
                        continue;
                    }

                    if (mlir::Value cond_value = translate_expression(TMP_CTX, reinterpret_cast<Expr*>(QUAL_NODE))) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully translated HAVING condition %d", i);
                        if (!cond_value.getType().isInteger(1)) {
                            cond_value = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), cond_value);
                        }

                        if (!predicate_result) {
                            predicate_result = cond_value;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first HAVING predicate");
                        } else {
                            predicate_result = predicate_builder.create<mlir::db::AndOp>(
                                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                                mlir::ValueRange{predicate_result, cond_value});
                            PGX_LOG(AST_TRANSLATE, DEBUG, "ANDed HAVING predicate %d", i);
                        }
                    } else {
                        PGX_WARNING("Failed to translate qual condition at index %d", i);
                    }
                }
            }
        }

        if (!predicate_result) {
            throw std::runtime_error("We parsed that there were predicates, but got nothing out of it!");
        }
        if (!predicate_result.getType().isInteger(1)) { // is boolean
            predicate_result = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                              predicate_result);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                         mlir::ValueRange{predicate_result});
    }

    TranslationResult result;
    result.op = selection_op;
    result.columns = input.columns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::apply_selection_from_qual_with_columns(const QueryCtxT& ctx,
                                                                           const TranslationResult& input,
                                                                           const List* qual) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 3] input: %s", input.toString().c_str());

    if (!input.op || !qual || qual->length == 0) {
        return input;
    }

    auto input_value = input.op->getResult(0);
    if (!input_value) {
        PGX_ERROR("Input operation has no result");
        throw std::runtime_error("Input operation has no result");
    }

    auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), input_value);

    {
        auto& predicate_region = selection_op.getPredicate();
        auto* predicate_block = new mlir::Block;
        predicate_region.push_back(predicate_block);

        const auto TUPLE_TYPE = mlir::relalg::TupleType::get(&context_);
        const auto TUPLE_ARG = predicate_block->addArgument(TUPLE_TYPE, ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicate_block);

        const auto TMP_CTX = QueryCtxT::createChildContext(ctx, predicate_builder, TUPLE_ARG);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu params", TMP_CTX.params.size());

        mlir::Value predicate_result = nullptr;
        if (qual && qual->length > 0) {
            if (!qual->elements) {
                PGX_WARNING("Qual list has length but no elements array - continuing without filter");
            } else {
                for (int i = 0; i < qual->length; i++) {
                    const ListCell* const lc = &qual->elements[i];
                    if (!lc) {
                        PGX_WARNING("Null ListCell at index %d", i);
                        continue;
                    }

                    auto *const QUAL_NODE = static_cast<Node*>(lfirst(lc));

                    if (!QUAL_NODE) {
                        PGX_WARNING("Null qual node at index %d", i);
                        continue;
                    }

                    mlir::Value cond_value;
                    cond_value = translate_expression(TMP_CTX, reinterpret_cast<Expr*>(QUAL_NODE));

                    if (cond_value) {
                        if (!cond_value.getType().isInteger(1)) {
                            cond_value = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), cond_value);
                        }

                        if (!predicate_result) {
                            predicate_result = cond_value;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first join predicate");
                        } else {
                            predicate_result = predicate_builder.create<mlir::db::AndOp>(
                                predicate_builder.getUnknownLoc(), mlir::ValueRange{predicate_result, cond_value});
                            PGX_LOG(AST_TRANSLATE, DEBUG, "ANDed join predicate %d", i);
                        }
                    } else {
                        PGX_WARNING("Failed to translate qual condition at index %d", i);
                    }
                }
            }
        }

        if (!predicate_result) {
            throw std::runtime_error("We parsed that there were predicates, but got nothing out of it!");
        }
        if (!predicate_result.getType().isInteger(1)) {
            predicate_result = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                              predicate_result);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                         mlir::ValueRange{predicate_result});
    }

    TranslationResult result;
    result.op = selection_op;
    result.columns = input.columns;
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 4] RESULT: %s", result.toString().c_str());
    return result;
}

auto PostgreSQLASTTranslator::Impl::build_scan_columns(List* targetlist,
                                                       const std::vector<pgx_lower::frontend::sql::ColumnInfo>& all_columns,
                                                       const std::string& table_scope) const
    -> std::vector<TranslationResult::ColumnSchema> {
    std::vector<TranslationResult::ColumnSchema> columns;
    const PostgreSQLTypeMapper TYPE_MAPPER(context_);

    ListCell* lc = nullptr;
    foreach (lc, targetlist) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle) {
            continue;
}
        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<Var*>(tle->expr);
            if (var->varattno > 0 && var->varattno <= static_cast<int>(all_columns.size())) {
                const auto& col_info = all_columns[var->varattno - 1];
                const mlir::Type MLIR_TYPE = TYPE_MAPPER.map_postgre_sqltype(col_info.type_oid, col_info.typmod,
                                                                            col_info.nullable);

                columns.push_back({.table_name = table_scope,
                                   .column_name = col_info.name,
                                   .type_oid = col_info.type_oid,
                                   .typmod = col_info.typmod,
                                   .mlir_type = MLIR_TYPE,
                                   .nullable = col_info.nullable});
            }
        }
    }
    return columns;
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_target_list(const QueryCtxT& ctx,
                                                                      const TranslationResult& input,
                                                                      const List* target_list,
                                                                      const TranslationResult* merged_join_child)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!input.op || !target_list || target_list->length <= 0 || !target_list->elements) {
        return input;
    }

    mlir::Value const input_value = input.op->getResult(0);
    if (!input_value) {
        PGX_ERROR("Input operation has no result");
        return input;
    }

    // When we have join context, we need to handle ALL target entries, not just computed ones
    // This ensures we project only the requested columns, not all input columns
    bool const handle_all_entries = (merged_join_child != nullptr);

    auto target_entries = std::vector<TargetEntry*>();
    auto computed_entries = std::vector<TargetEntry*>();

    for (int i = 0; i < target_list->length; i++) {
        auto* tle = static_cast<TargetEntry*>(lfirst(&target_list->elements[i]));
        if (tle) {
            if (handle_all_entries) {
                target_entries.push_back(tle);
                if (tle->expr && tle->expr->type != T_Var) {
                    computed_entries.push_back(tle);
                }
            } else {
                target_entries.push_back(tle);
                if (tle->expr && tle->expr->type != T_Var) {
                    computed_entries.push_back(tle);
                }
            }
        }
    }

    if (!handle_all_entries && computed_entries.empty()) {
        return input;
    }

    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    // First pass: Translate expressions to get their types. We need a tuple context, so create a temporary MapOp
    auto expression_types = std::vector<mlir::Type>();
    auto column_names = std::vector<std::string>();
    auto expression_oids = std::vector<Oid>();
    {
        auto placeholder_attrs = std::vector<mlir::Attribute>();
        for (auto i = 0; i < computed_entries.size(); i++) {
            auto temp_name = std::string("temp_") + std::to_string(i);
            auto attr = column_manager.createDef(COMPUTED_EXPRESSION_SCOPE, temp_name);
            attr.getColumn().type = mlir::NoneType::get(&context_);
            placeholder_attrs.push_back(attr);
        }

        auto temp_map_op = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), input_value,
                                                                 ctx.builder.getArrayAttr(placeholder_attrs));

        {
            auto& temp_region = temp_map_op.getPredicate();
            auto* temp_block = &temp_region.emplaceBlock();
            auto tuple_arg = temp_block->addArgument(mlir::relalg::TupleType::get(&context_), ctx.builder.getUnknownLoc());

            mlir::OpBuilder temp_builder(&context_);
            temp_builder.setInsertionPointToStart(temp_block);
            auto tmp_ctx = QueryCtxT::createChildContext(ctx, temp_builder, tuple_arg);

            for (auto* entry : computed_entries) {
                auto col_name = entry->resname ? entry->resname : "col_" + std::to_string(entry->resno);
                if (col_name == "?column?") {
                    col_name = "col_" + std::to_string(entry->resno);
}

                // TODO: NV: This is bad. This should be using the TranslationResult to find the name. Actually, most of
                // this function
                //           seems kind of bad to me. But oh well, it's working for now I guess.
                if (!entry->resname && ctx.current_stmt.planTree) {
                    const Plan* const top_plan = ctx.current_stmt.planTree;
                    const Agg* agg_node = nullptr;

                    if (top_plan->type == T_Agg) {
                        agg_node = reinterpret_cast<const Agg*>(top_plan);
                    } else if (top_plan->type == T_Sort && top_plan->lefttree && top_plan->lefttree->type == T_Agg) {
                        agg_node = reinterpret_cast<const Agg*>(top_plan->lefttree);
                    }

                    if (agg_node && agg_node->plan.targetlist) {
                        ListCell* lc = nullptr;
                        int idx = 0;
                        foreach (lc, agg_node->plan.targetlist) {
                            idx++;
                            if (idx == entry->resno) {
                                const auto* agg_te = static_cast<const TargetEntry*>(lfirst(lc));
                                if (agg_te->resname) {
                                    col_name = agg_te->resname;
                                    PGX_LOG(AST_TRANSLATE, DEBUG,
                                            "MapOp: Using name '%s' from parent Agg's targetlist for expression",
                                            col_name.c_str());
                                }
                                break;
                            }
                        }
                    }
                }

                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "MapOp: Creating computed column '%s' from targetentry resno=%d resname='%s'", col_name.c_str(),
                        entry->resno, entry->resname ? entry->resname : "<null>");

                if (mlir::Value const expr_value = translate_expression(tmp_ctx, entry->expr)) {
                    mlir::Type const expr_mlir_type = expr_value.getType();
                    expression_types.push_back(expr_mlir_type);
                    column_names.push_back(col_name);
                    Oid const type_oid = PostgreSQLTypeMapper::map_mlir_type_to_oid(expr_mlir_type);
                    expression_oids.push_back(type_oid);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "MapOp column '%s': MLIR type mapped to OID=%u",
                            col_name.c_str(), type_oid);
                } else {
                    PGX_WARNING("Failed to get expression!!");
                }
            }
        }
        temp_map_op.erase();

        if (expression_types.empty()) {
            return input;
        }
    }

    mlir::relalg::MapOp map_op;
    {
        std::vector<mlir::Attribute> computed_col_attrs;
        for (size_t i = 0; i < expression_types.size(); i++) {
            auto column_ptr = column_manager.get(COMPUTED_EXPRESSION_SCOPE, column_names[i]);
            column_ptr->type = expression_types[i];
            computed_col_attrs.push_back(column_manager.createDef(COMPUTED_EXPRESSION_SCOPE, column_names[i]));
        }

        map_op = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), input_value,
                                                        ctx.builder.getArrayAttr(computed_col_attrs));

        // Build computation region
        auto& predicate_region = map_op.getPredicate();
        auto* predicate_block = new mlir::Block;
        predicate_region.push_back(predicate_block);
        auto tuple_arg = predicate_block->addArgument(mlir::relalg::TupleType::get(&context_), ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicate_block);
        auto tmp_ctx = QueryCtxT::createChildContext(ctx, predicate_builder, tuple_arg);

        std::vector<mlir::Value> computed_values;
        for (auto* entry : computed_entries) {
            mlir::Value expr_value;
            if (merged_join_child != nullptr) {
                expr_value = translate_expression(tmp_ctx, entry->expr);
            } else {
                expr_value = translate_expression(tmp_ctx, entry->expr);
            }

            if (expr_value) {
                computed_values.push_back(expr_value);
            } else {
                PGX_WARNING("Failed to get expression!!");
            }
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), computed_values);
    }

    std::vector<TranslationResult::ColumnSchema> all_columns = input.columns;
    for (size_t i = 0; i < expression_types.size(); i++) {
        all_columns.push_back({.table_name = COMPUTED_EXPRESSION_SCOPE,
                             .column_name = column_names[i],
                             .type_oid = expression_oids[i],
                             .typmod = -1,
                             .mlir_type = expression_types[i],
                             .nullable = true});
    }

    TranslationResult intermediate_result;
    intermediate_result.op = map_op;

    if (!handle_all_entries) {
        // Build result columns in TARGETLIST ORDER (not input-first order)
        // Include ALL entries (both resjunk and non-resjunk) for downstream operations like Sort
        size_t computed_idx = 0;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Building columns from %zu targetEntries, input has %zu columns, %zu computed",
                target_entries.size(), input.columns.size(), expression_types.size());
        for (size_t i = 0; i < target_entries.size(); i++) {
            auto* tle = target_entries[i];

            PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] resno=%d, resjunk=%d, expr type=%d",
                    i, tle->resno, tle->resjunk, tle->expr ? tle->expr->type : -1);

            // Include ALL targetlist entries (don't skip resjunk here - materialize will filter)
            if (tle->expr && tle->expr->type == T_Var) {
                // This is a Var - find it in input.columns by name, not by varattno
                const auto* var = reinterpret_cast<const Var*>(tle->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "    Var: varno=%d, varattno=%d", var->varno, var->varattno);

                // Resolve the Var to get table and column name
                std::string tableName;
                std::string colName;
                bool nullable = false;

                std::optional<int> const varnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varnosyn) : std::nullopt;
                std::optional<int> const varattnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varattnosyn) : std::nullopt;

                if (auto resolved = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt)) {
                    tableName = resolved->table_name;
                    colName = resolved->column_name;
                    nullable = resolved->nullable;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Resolved via varno_resolution to %s.%s", tableName.c_str(), colName.c_str());
                } else {
                    // Fallback: use PostgreSQL catalog
                    int const schema_varno = IS_SPECIAL_VARNO(var->varno) ? var->varnosyn : var->varno;
                    tableName = get_table_alias_from_rte(&ctx.current_stmt, schema_varno);
                    colName = get_column_name_from_schema(&ctx.current_stmt, schema_varno, var->varattno);
                    nullable = is_column_nullable(&ctx.current_stmt, schema_varno, var->varattno);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Resolved via schema lookup to %s.%s", tableName.c_str(), colName.c_str());
                }

                // Find the column in input.columns by matching table and column name
                bool found = false;
                for (const auto& col : input.columns) {
                    if (col.table_name == tableName && col.column_name == colName) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "    Adding from input: %s.%s", col.table_name.c_str(), col.column_name.c_str());
                        intermediate_result.columns.push_back(col);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Column %s.%s not found in input", tableName.c_str(), colName.c_str());
                }
            } else {
                // This is a computed expression - use the computedIdx
                PGX_LOG(AST_TRANSLATE, DEBUG, "    Computed expr, computedIdx=%zu", computed_idx);
                if (computed_idx < expression_types.size()) {
                    size_t const column_index = input.columns.size() + computed_idx;
                    const auto& col = all_columns[column_index];
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Adding computed: %s.%s", col.table_name.c_str(), col.column_name.c_str());
                    intermediate_result.columns.push_back(col);
                    computed_idx++;
                }
            }
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "apply_projection returning intermediateResult with %zu columns (handleAllEntries=false, includes resjunk):", intermediate_result.columns.size());
        for (size_t i = 0; i < intermediate_result.columns.size(); i++) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s", i, intermediate_result.columns[i].table_name.c_str(), intermediate_result.columns[i].column_name.c_str());
        }
        return intermediate_result;
    }

    intermediate_result.columns = all_columns;

    // When handling all entries we need to add a ProjectionOp to select only the columns from the target list
    std::vector<mlir::Attribute> projected_column_refs;
    std::vector<TranslationResult::ColumnSchema> projected_columns;

    size_t computed_idx = 0;
    for (auto* tle : target_entries) {
        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<const Var*>(tle->expr);

            size_t const left_column_count = merged_join_child ? merged_join_child->left_child_column_count : 0;
            size_t const right_column_count = merged_join_child ? (merged_join_child->columns.size() - left_column_count) : 0;
            bool const input_contains_both_sides = (merged_join_child != nullptr)
                                          && (input.columns.size() >= merged_join_child->columns.size());

            size_t column_index = SIZE_MAX;
            if (var->varno == OUTER_VAR && merged_join_child && left_column_count > 0) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(left_column_count)) {
                    column_index = var->varattno - 1;
                }
            } else if (var->varno == INNER_VAR && merged_join_child && right_column_count > 0) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(right_column_count)) {
                    if (input_contains_both_sides) {
                        column_index = left_column_count + (var->varattno - 1);
                    } else {
                        column_index = var->varattno - 1;
                    }
                }
            } else if (var->varattno > 0 && var->varattno <= static_cast<int>(input.columns.size())) {
                column_index = var->varattno - 1;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving Var (varno=%d, varattno=%d) to input column %zu: %s.%s",
                        var->varno, var->varattno, column_index, input.columns[column_index].table_name.c_str(),
                        input.columns[column_index].column_name.c_str());
            } else {
                throw std::runtime_error("Failed");
            }

            if (column_index < intermediate_result.columns.size()) {
                const auto& col = intermediate_result.columns[column_index];
                auto col_ref = column_manager.createRef(col.table_name, col.column_name);
                projected_column_refs.push_back(col_ref);
                projected_columns.push_back(col);
            }
        } else {
            size_t const column_index = input.columns.size() + computed_idx;
            if (column_index < intermediate_result.columns.size()) {
                const auto& col = intermediate_result.columns[column_index];
                auto col_ref = column_manager.createRef(col.table_name, col.column_name);
                projected_column_refs.push_back(col_ref);
                projected_columns.push_back(col);
                computed_idx++;
            }
        }
    }

    // Create ProjectionOp
    auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    const auto PROJECTION_OP = ctx.builder.create<mlir::relalg::ProjectionOp>(
        ctx.builder.getUnknownLoc(), tuple_stream_type,
        mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all), map_op.getResult(),
        ctx.builder.getArrayAttr(projected_column_refs));

    TranslationResult result;
    result.op = PROJECTION_OP;
    result.columns = projected_columns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_translation_result(
    const QueryCtxT& ctx, const TranslationResult& input, const TranslationResult& merged_join_child,
    const List* target_list, const JoinType JOIN_TYPE) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    if (!input.op || !target_list || target_list->length <= 0) {
        PGX_WARNING("No target list");
        return input;
    }

    auto input_value = input.op->getResult(0);
    if (!input_value) {
        PGX_ERROR("Input operation has no result");
        return input;
    }

    std::vector<TranslationResult::ColumnSchema> projected_columns;
    std::vector<mlir::Attribute> column_refs;
    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    size_t const left_column_count = merged_join_child.left_child_column_count;
    size_t const right_column_count = merged_join_child.columns.size() - left_column_count;

    ListCell* lc = nullptr;
    foreach (lc, target_list) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle) {
            continue;
}

        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<Var*>(tle->expr);
            size_t column_index = SIZE_MAX;

            if (var->varno == OUTER_VAR) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(left_column_count)) {
                    column_index = var->varattno - 1;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: OUTER_VAR varattno=%d maps to position %zu",
                            var->varattno, column_index);
                }
            } else if (var->varno == INNER_VAR) {
                const bool IS_EXISTS_JOIN = (JOIN_TYPE == JOIN_SEMI || JOIN_TYPE == JOIN_ANTI
                                             || JOIN_TYPE == JOIN_RIGHT_ANTI);
                if (IS_EXISTS_JOIN) {
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Projection: Skipping INNER_VAR reference after %s join (right columns not in output)",
                            JOIN_TYPE == JOIN_SEMI ? "SEMI" : (JOIN_TYPE == JOIN_ANTI ? "ANTI" : "RIGHT_ANTI"));
                    continue;
                }

                std::optional<int> const varnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varnosyn)
                                                                               : std::nullopt;
                std::optional<int> const varattnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varattnosyn)
                                                                                  : std::nullopt;

                if (auto mapping = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt)) {
                    const auto& table_name = mapping->table_name;
                    const auto& col_name = mapping->column_name;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR using varno_resolution -> @%s::@%s",
                            table_name.c_str(), col_name.c_str());

                    for (size_t i = 0; i < input.columns.size(); ++i) {
                        if (input.columns[i].table_name == table_name && input.columns[i].column_name == col_name) {
                            column_index = i;
                            break;
                        }
                    }
                }

                if (column_index == SIZE_MAX && var->varattno > 0 && var->varattno <= static_cast<int>(right_column_count))
                {
                    column_index = left_column_count + (var->varattno - 1);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR varattno=%d maps to position %zu (fallback)",
                            var->varattno, column_index);
                }
            } else {
                PGX_WARNING("Unexpected varno %d in join projection", var->varno);
                continue;
            }

            if (column_index < input.columns.size()) {
                const auto& col = input.columns[column_index];
                projected_columns.push_back(col);

                auto col_ref = column_manager.createRef(col.table_name, col.column_name);
                column_refs.push_back(col_ref);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Projecting column: %s.%s from position %zu", col.table_name.c_str(),
                        col.column_name.c_str(), column_index);
            } else {
                PGX_ERROR("Column index %zu out of bounds (have %zu columns)", column_index, input.columns.size());
            }
        } else if (tle->expr) {
            PGX_LOG(AST_TRANSLATE, DEBUG,
                    "Non-Var expression in join projection, delegating to apply_projection_from_target_list");
            auto result = apply_projection_from_target_list(ctx, input, target_list, &merged_join_child);
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
            return result;
        }
    }

    const auto COLUMNS_IDENTICAL = [&]() {
        if (projected_columns.size() != input.columns.size()) {
            return false;
}
        for (size_t i = 0; i < projected_columns.size(); ++i) {
            if (projected_columns[i].table_name != input.columns[i].table_name
                || projected_columns[i].column_name != input.columns[i].column_name) {
                return false;
}
        }
        return true;
    };

    if (!projected_columns.empty() && !COLUMNS_IDENTICAL()) {
        auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
        const auto PROJECTION_OP = ctx.builder.create<mlir::relalg::ProjectionOp>(
            ctx.builder.getUnknownLoc(), tuple_stream_type,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all), input_value,
            ctx.builder.getArrayAttr(column_refs));

        TranslationResult result;
        result.op = PROJECTION_OP;
        result.columns = projected_columns;

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created ProjectionOp: projecting %zu columns from %zu input columns",
                projected_columns.size(), input.columns.size());
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
        return result;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] No projection needed - columns already in correct order");
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", input.toString().c_str());
    return input;
}

auto PostgreSQLASTTranslator::Impl::create_materialize_op(const QueryCtxT& context, const mlir::Value TUPLE_STREAM,
                                                          const TranslationResult& translation_result) const
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!translation_result.columns.empty()) {
        auto& column_manager = context.builder.getContext()
                                  ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                                  ->getColumnManager();
        std::vector<mlir::Attribute> column_ref_attrs;
        std::vector<mlir::Attribute> column_name_attrs;

        const auto* top_plan = context.current_stmt.planTree;
        const auto* target_list = top_plan ? top_plan->targetlist : nullptr;

        for (size_t col_index = 0; col_index < translation_result.columns.size(); col_index++) {
            const auto& column = translation_result.columns[col_index];
            if (target_list && col_index < static_cast<size_t>(list_length(target_list))) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(target_list, col_index));
                if (tle && tle->resjunk) {
                    continue;
                }
            }

            auto output_name = column.column_name;

            if (target_list && col_index < static_cast<size_t>(list_length(target_list))) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(target_list, col_index));
                if (tle && tle->resname) {
                    output_name = tle->resname;
                }
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "MaterializeOp column %zu: %s.%s -> output name '%s'", col_index,
                    column.table_name.c_str(), column.column_name.c_str(), output_name.c_str());

            auto col_ref = column_manager.createRef(column.table_name, column.column_name);
            column_ref_attrs.push_back(col_ref);

            auto name_attr = context.builder.getStringAttr(output_name);
            column_name_attrs.push_back(name_attr);
        }

        auto column_refs = context.builder.getArrayAttr(column_ref_attrs);
        auto column_names = context.builder.getArrayAttr(column_name_attrs);
        auto table_type = mlir::dsa::TableType::get(&context_);

        auto materialize_op = context.builder.create<mlir::relalg::MaterializeOp>(
            context.builder.getUnknownLoc(), table_type, TUPLE_STREAM, column_refs, column_names);
        return materialize_op.getResult();
    }         throw std::runtime_error("Should be impossible");
   
    return {};
}

auto PostgreSQLASTTranslator::Impl::merge_translation_results(const TranslationResult* left_child,
                                                              const TranslationResult* right_child) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    TranslationResult merged_result;

    size_t left_size = 0;

    if (left_child) {
        left_size = left_child->columns.size();
        merged_result.columns.insert(merged_result.columns.end(), left_child->columns.begin(), left_child->columns.end());

        PGX_LOG(AST_TRANSLATE, DEBUG, "Merged %zu columns from left_child", left_child->columns.size());
    }

    if (right_child) {
        merged_result.columns.insert(merged_result.columns.end(), right_child->columns.begin(),
                                     right_child->columns.end());

        PGX_LOG(AST_TRANSLATE, DEBUG, "Merged %zu columns from right_child", right_child->columns.size());
    }

    merged_result.left_child_column_count = left_size;

    PGX_LOG(AST_TRANSLATE, DEBUG,
            "merge_translation_results: Total %zu columns (%zu left + %zu right), ??? varno mappings",
            merged_result.columns.size(), left_size, merged_result.columns.size() - left_size);

    return merged_result;
}

auto create_child_context_with_var_mappings(
    const QueryCtxT& parent, const std::map<std::pair<int, int>, std::pair<std::string, std::string>>& var_mappings)
    -> QueryCtxT {
    auto child_ctx = QueryCtxT::createChildContext(parent);

    PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR_MAPPINGS] Creating child context, parent had %zu varno_resolution entries",
            parent.varno_resolution.size());

    std::erase_if(child_ctx.varno_resolution, [](const auto& entry) {
        const auto& [varno, varattno] = entry.first;
        return varno == INNER_VAR || varno == OUTER_VAR;
    });

    for (const auto& [key, value] : var_mappings) {
        const auto& [varno, varattno] = key;
        if (varno != INNER_VAR && varno != OUTER_VAR) {
            PGX_WARNING("create_child_context_with_var_mappings: var_mappings contains varno=%d (expected -1 or -2)",
                        varno);
            continue;
        }
        child_ctx.varno_resolution[key] = value;
        PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR_MAPPINGS] Added mapping: varno=%d, varattno=%d -> (%s, %s)", varno,
                varattno, value.first.c_str(), value.second.c_str());
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR_MAPPINGS] Final context has %zu varno_resolution entries",
            child_ctx.varno_resolution.size());

    for (const auto& [key, value] : child_ctx.varno_resolution) {
        const auto& [varno, varattno] = key;
        if (varno == INNER_VAR || varno == OUTER_VAR) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR_MAPPINGS] Final mapping: varno=%d, varattno=%d -> (%s, %s)", varno,
                    varattno, value.first.c_str(), value.second.c_str());
        }
    }

    return child_ctx;
}

auto map_child_cols(const QueryCtxT& ctx, const TranslationResult* left_translation,
                    const TranslationResult* right_translation) -> QueryCtxT {
    std::map<std::pair<int, int>, std::pair<std::string, std::string>> child_mappings;

    if (left_translation) {
        for (size_t i = 0; i < left_translation->columns.size(); ++i) {
            child_mappings[{OUTER_VAR, i + 1}] = {left_translation->columns[i].table_name,
                                                  left_translation->columns[i].column_name};
        }
    }

    if (right_translation) {
        for (size_t i = 0; i < right_translation->columns.size(); ++i) {
            child_mappings[{INNER_VAR, i + 1}] = {right_translation->columns[i].table_name,
                                                  right_translation->columns[i].column_name};
        }
    }

    auto child_ctx = create_child_context_with_var_mappings(ctx, child_mappings);
    if (left_translation) {
        child_ctx.outer_result = std::ref(*left_translation);
    }

    return child_ctx;
}

} // namespace postgresql_ast
