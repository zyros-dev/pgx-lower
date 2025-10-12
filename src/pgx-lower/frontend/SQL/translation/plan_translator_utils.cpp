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
#include "utils/memutils.h"  // CurTransactionContext
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

    const size_t params_before = ctx.params.size();
    process_init_plans(ctx, plan);
    PGX_LOG(AST_TRANSLATE, DEBUG, "After processing InitPlans: context has %zu InitPlans (%zu new)", ctx.params.size(),
            ctx.params.size() - params_before);

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

    auto childResult = translate_plan_node(ctx, sort->plan.lefttree);
    if (!childResult.op) {
        PGX_ERROR("Failed to translate Sort child plan");
        return childResult;
    }
    PGX_LOG(AST_TRANSLATE, DEBUG, "Sort node got %s", childResult.toString().data());

    if (!sort->numCols || !sort->sortColIdx) {
        return childResult;
    }

    auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    std::vector<mlir::Attribute> sortSpecs;
    for (int i = 0; i < sort->numCols; i++) {
        const AttrNumber colIdx = sort->sortColIdx[i];
        if (colIdx <= 0 || colIdx >= MAX_COLUMN_INDEX)
            continue;

        auto spec = mlir::relalg::SortSpec::asc;
        if (sort->sortOperators) {
            if (char* oprname = get_opname(sort->sortOperators[i])) {
                spec = (std::string(oprname) == ">" || std::string(oprname) == ">=") ? mlir::relalg::SortSpec::desc
                                                                                     : mlir::relalg::SortSpec::asc;
                pfree(oprname);
            }
        }

        ListCell* lc;
        int idx = 0;
        foreach (lc, sort->plan.targetlist) {
            if (++idx != colIdx)
                continue;

            const TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (IsA(tle->expr, Var)) {
                const Var* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= childResult.columns.size()) {
                    const auto& column = childResult.columns[var->varattno - 1];
                    sortSpecs.push_back(mlir::relalg::SortSpecificationAttr::get(
                        ctx.builder.getContext(), columnManager.createRef(column.table_name, column.column_name), spec));
                }
            }
            break;
        }
    }

    if (sortSpecs.empty()) {
        return childResult;
    }

    // Allocate SortSpecification struct in long-lived PostgreSQL memory context
    // Must survive from AST translation through MLIR compilation to runtime execution
    const int32_t numColumns = childResult.columns.size();
    const int32_t numSortKeys = sort->numCols;

    // Use CurTransactionContext for long-lived allocation (declared in PostgreSQL headers)
    MemoryContext oldContext = MemoryContextSwitchTo(CurTransactionContext);

    // Allocate column info array
    auto* colInfos = static_cast<runtime::SortColumnInfo*>(
        MemoryContextAlloc(CurTransactionContext, numColumns * sizeof(runtime::SortColumnInfo))
    );

    for (int i = 0; i < numColumns; i++) {
        colInfos[i].table_name = pstrdup(childResult.columns[i].table_name.c_str());  // pstrdup already uses current context
        colInfos[i].column_name = pstrdup(childResult.columns[i].column_name.c_str());
        colInfos[i].type_oid = childResult.columns[i].type_oid;
        colInfos[i].typmod = childResult.columns[i].typmod;
    }

    // Allocate and fill sort key arrays
    auto* sortKeyIdxs = static_cast<int32_t*>(MemoryContextAlloc(CurTransactionContext, numSortKeys * sizeof(int32_t)));
    auto* sortOps = static_cast<uint32_t*>(MemoryContextAlloc(CurTransactionContext, numSortKeys * sizeof(uint32_t)));
    auto* collations = static_cast<uint32_t*>(MemoryContextAlloc(CurTransactionContext, numSortKeys * sizeof(uint32_t)));
    auto* nullsFirst = static_cast<bool*>(MemoryContextAlloc(CurTransactionContext, numSortKeys * sizeof(bool)));

    int validSortKeys = 0;
    for (int i = 0; i < numSortKeys; i++) {
        const AttrNumber colIdx = sort->sortColIdx[i];
        if (colIdx <= 0 || colIdx > numColumns)
            continue;

        sortKeyIdxs[validSortKeys] = colIdx - 1; // 0-based
        sortOps[validSortKeys] = sort->sortOperators ? sort->sortOperators[i] : InvalidOid;
        collations[validSortKeys] = sort->collations ? sort->collations[i] : InvalidOid;
        nullsFirst[validSortKeys] = sort->nullsFirst ? sort->nullsFirst[i] : false;
        validSortKeys++;
    }

    if (validSortKeys == 0) {
        // Free allocated memory if no valid sort keys (still in transaction context)
        pfree(colInfos);
        pfree(sortKeyIdxs);
        pfree(sortOps);
        pfree(collations);
        pfree(nullsFirst);
        MemoryContextSwitchTo(oldContext);  // Restore original context
        return childResult;
    }

    // Allocate spec struct
    auto* spec = static_cast<runtime::SortSpecification*>(
        MemoryContextAlloc(CurTransactionContext, sizeof(runtime::SortSpecification))
    );
    spec->columns = colInfos;
    spec->num_columns = numColumns;
    spec->sort_key_indices = sortKeyIdxs;
    spec->sort_operators = sortOps;
    spec->collations = collations;
    spec->nulls_first = nullsFirst;
    spec->num_sort_keys = validSortKeys;

    // Restore original memory context
    MemoryContextSwitchTo(oldContext);

    // Create attribute with pointer to spec (using IntegerAttr for simplicity)
    // Use signless i64 type required by MLIR arith.constant
    auto specPtrAttr = ctx.builder.getIntegerAttr(
        ctx.builder.getIntegerType(64),  // signless 64-bit
        reinterpret_cast<uint64_t>(spec)
    );

    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    const auto sortOp = ctx.builder.create<mlir::relalg::SortOp>(
        ctx.builder.getUnknownLoc(), tupleStreamType, childResult.op->getResult(0), ctx.builder.getArrayAttr(sortSpecs), specPtrAttr);

    TranslationResult result;
    result.op = sortOp;

    if (sort->plan.targetlist) {
        result.columns.clear();
        ListCell* lc;
        foreach (lc, sort->plan.targetlist) {
            const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (!tle)
                continue;

            if (tle->expr && IsA(tle->expr, Var)) {
                const auto* var = reinterpret_cast<Var*>(tle->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "Sort targetentry: resjunk=%d, varattno=%d, childResult.columns.size()=%zu",
                        tle->resjunk, var->varattno, childResult.columns.size());
                if (var->varattno > 0 && var->varattno <= childResult.columns.size()) {
                    const auto& col = childResult.columns[var->varattno - 1];
                    PGX_LOG(AST_TRANSLATE, DEBUG, "  Adding column: %s.%s", col.table_name.c_str(), col.column_name.c_str());
                    result.columns.push_back(col);
                }
            }
        }
    } else {
        result.columns = childResult.columns;
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

    TranslationResult childResult;

    if (Plan* leftTree = limit->plan.lefttree) {
        childResult = translate_plan_node(ctx, leftTree);
        if (!childResult.op) {
            PGX_ERROR("Failed to translate Limit child plan");
            return childResult;
        }
    } else {
        PGX_WARNING("Limit node has no child plan");
        return TranslationResult{};
    }

    auto childOutput = childResult.op->getResult(0);
    if (!childOutput) {
        PGX_ERROR("Child operation has no result");
        return childResult;
    }

    int64_t limitCount = DEFAULT_LIMIT_COUNT;
    int64_t limitOffset = 0;

    Node* limitOffsetNode = limit->limitOffset;

    if (Node* limitCountNode = limit->limitCount) {
        Node* node = limitCountNode;
        if (IsA(node, Const)) {
            const Const* constNode = reinterpret_cast<Const*>(node);
            if (!constNode->constisnull) {
                limitCount = static_cast<int64_t>(constNode->constvalue);
            }
        } else {
            PGX_WARNING("Limit count is not a Const or Param node");
        }
    }

    if (limitOffsetNode) {
        Node* node = limitOffsetNode;
        if (IsA(node, Const)) {
            const Const* constNode = reinterpret_cast<Const*>(node);
            if (!constNode->constisnull) {
                limitOffset = static_cast<int64_t>(constNode->constvalue);
            }
        }
    }

    if (limitCount < 0) {
        PGX_WARNING("Invalid negative limit count: %d", limitCount);
        limitCount = DEFAULT_LIMIT_COUNT;
    } else if (limitCount > MAX_LIMIT_COUNT) {
        PGX_WARNING("Very large limit count: %d", limitCount);
    }

    if (limitOffset < 0) {
        PGX_WARNING("Negative offset not supported, using 0");
        limitOffset = 0;
    }

    if (limitCount == -1) {
        limitCount = INT32_MAX;
    }

    const auto limitOp = ctx.builder.create<mlir::relalg::LimitOp>(
        ctx.builder.getUnknownLoc(), ctx.builder.getI32IntegerAttr(static_cast<int32_t>(limitCount)), childOutput);

    TranslationResult result;
    result.op = limitOp;
    result.columns = childResult.columns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_gather(QueryCtxT& ctx, const Gather* gather) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!gather) {
        PGX_ERROR("Invalid Gather parameters");
        return TranslationResult{};
    }

    TranslationResult childResult;

    if (Plan* leftTree = gather->plan.lefttree) {
        childResult = translate_plan_node(ctx, leftTree);
        if (!childResult.op) {
            PGX_ERROR("Failed to translate Gather child plan");
            return childResult;
        }
    } else {
        PGX_WARNING("Gather node has no child plan");
        return TranslationResult{};
    }

    return childResult;
}

auto PostgreSQLASTTranslator::Impl::translate_gather_merge(QueryCtxT& ctx, const GatherMerge* gatherMerge)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!gatherMerge) {
        PGX_ERROR("Invalid GatherMerge parameters");
        return TranslationResult{};
    }

    TranslationResult childResult;
    if (Plan* leftTree = gatherMerge->plan.lefttree) {
        childResult = translate_plan_node(ctx, leftTree);
        if (!childResult.op) {
            PGX_ERROR("Failed to translate GatherMerge child plan");
            return childResult;
        }
    } else {
        PGX_WARNING("GatherMerge node has no child plan");
        return TranslationResult{};
    }

    // GatherMerge is parallel execution coordinator - pass through child for now
    // Note: Ignoring sort columns (numCols, sortColIdx, etc.) as child already sorted
    PGX_LOG(AST_TRANSLATE, DEBUG, "GatherMerge: passing through child result (parallel gathering not implemented)");
    return childResult;
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

    const List* all_subplans = ctx.current_stmt.subplans;
    const int num_subplans = list_length(all_subplans);
    ListCell* lc;
    foreach (lc, plan->initPlan) {
        const auto* subplan = static_cast<SubPlan*>(lfirst(lc));
        if (!subplan) {
            PGX_ERROR("Invalid SubPlan in initPlan list");
            continue;
        }

        const int plan_id = subplan->plan_id;
        if (plan_id < 1 || plan_id > num_subplans) {
            PGX_ERROR("SubPlan plan_id %d out of range (have %d subplans)", plan_id, num_subplans);
            continue;
        }

        auto* initplan = static_cast<Plan*>(list_nth(all_subplans, plan_id - 1));
        if (!initplan) {
            PGX_ERROR("SubPlan plan_id %d points to null Plan", plan_id);
            continue;
        }

        auto initplan_result = translate_plan_node(ctx, initplan);
        if (!initplan_result.op) {
            PGX_ERROR("Failed to translate InitPlan (plan_id=%d)", plan_id);
            continue;
        }

        const List* setParam = subplan->setParam;
        if (!setParam || list_length(setParam) == 0) {
            PGX_ERROR("InitPlan has no setParam");
            continue;
        }
        const int paramid = list_nth_int(setParam, 0);

        if (initplan_result.columns.empty()) {
            PGX_ERROR("InitPlan produced no columns for paramid=%d", paramid);
            continue;
        }

        const bool is_cte = (subplan->subLinkType == 7) || (initplan_result.columns.size() > 1);
        if (is_cte) {
            ctx.initplan_results[paramid] = initplan_result;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Stored CTE InitPlan result for paramid=%d (plan_id=%d, %zu columns)",
                    paramid, plan_id, initplan_result.columns.size());
        } else {
            const auto& col = initplan_result.columns[0];
            ctx.params[paramid] = pgx_lower::frontend::sql::ResolvedParam{
                .table_name = col.table_name,
                .column_name = col.column_name,
                .type_oid = col.type_oid,
                .typmod = col.typmod,
                .nullable = col.nullable,
                .mlir_type = col.mlir_type,
                .cached_value = initplan_result.op->getResult(0)};
            PGX_LOG(AST_TRANSLATE, DEBUG, "Stored scalar InitPlan result for paramid=%d (plan_id=%d)", paramid, plan_id);
        }
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Processed %d InitPlans, context now has %zu total", list_length(plan->initPlan),
            ctx.params.size());
}

auto PostgreSQLASTTranslator::Impl::create_query_function(mlir::OpBuilder& builder) -> mlir::func::FuncOp {
    PGX_IO(AST_TRANSLATE);
    auto tableType = mlir::dsa::TableType::get(builder.getContext());
    auto queryFuncType = builder.getFunctionType({}, {tableType});
    auto queryFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), QUERY_FUNCTION_NAME, queryFuncType);

    auto& queryBody = queryFunc.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&queryBody);

    return queryFunc;
}

auto PostgreSQLASTTranslator::Impl::apply_selection_from_qual(const QueryCtxT& ctx, const TranslationResult& input,
                                                              const List* qual) -> TranslationResult {
    // This applies a filter to the SELECT statement, so it can be a SELECT x FROM y WHERE z, or a
    // GROUP BY HAVING statement for instance.
    PGX_IO(AST_TRANSLATE);
    if (!input.op || !qual || qual->length == 0) {
        return input;
    }

    auto inputValue = input.op->getResult(0);
    if (!inputValue) {
        PGX_ERROR("Input operation has no result");
        throw std::runtime_error("Input operation has no result");
    }

    auto selectionOp = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), inputValue);

    { // Build the predicate region
        auto& predicateRegion = selectionOp.getPredicate();
        auto* predicateBlock = new mlir::Block;
        predicateRegion.push_back(predicateBlock);

        const auto tupleType = mlir::relalg::TupleType::get(&context_);
        const auto tupleArg = predicateBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicateBlock);

        const auto tmp_ctx = QueryCtxT::createChildContext(ctx, predicate_builder, tupleArg);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu params", tmp_ctx.params.size());

        mlir::Value predicateResult = nullptr;
        if (qual && qual->length > 0) {
            if (!qual->elements) {
                PGX_WARNING("Qual list has length but no elements array - continuing without filter");
            } else {
                for (int i = 0; i < qual->length; i++) {
                    const ListCell* lc = &qual->elements[i];
                    if (!lc) {
                        PGX_WARNING("Null ListCell at index %d", i);
                        continue;
                    }

                    const auto qualNode = static_cast<Node*>(lfirst(lc));

                    if (!qualNode) {
                        PGX_WARNING("Null qual node at index %d", i);
                        continue;
                    }

                    if (mlir::Value condValue = translate_expression(tmp_ctx, reinterpret_cast<Expr*>(qualNode))) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully translated HAVING condition %d", i);
                        if (!condValue.getType().isInteger(1)) {
                            condValue = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), condValue);
                        }

                        if (!predicateResult) {
                            predicateResult = condValue;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first HAVING predicate");
                        } else {
                            predicateResult = predicate_builder.create<mlir::db::AndOp>(
                                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                                mlir::ValueRange{predicateResult, condValue});
                            PGX_LOG(AST_TRANSLATE, DEBUG, "ANDed HAVING predicate %d", i);
                        }
                    } else {
                        PGX_WARNING("Failed to translate qual condition at index %d", i);
                    }
                }
            }
        }

        if (!predicateResult) {
            throw std::runtime_error("We parsed that there were predicates, but got nothing out of it!");
        }
        if (!predicateResult.getType().isInteger(1)) { // is boolean
            predicateResult = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                              predicateResult);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                         mlir::ValueRange{predicateResult});
    }

    TranslationResult result;
    result.op = selectionOp;
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

    auto inputValue = input.op->getResult(0);
    if (!inputValue) {
        PGX_ERROR("Input operation has no result");
        throw std::runtime_error("Input operation has no result");
    }

    auto selectionOp = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), inputValue);

    {
        auto& predicateRegion = selectionOp.getPredicate();
        auto* predicateBlock = new mlir::Block;
        predicateRegion.push_back(predicateBlock);

        const auto tupleType = mlir::relalg::TupleType::get(&context_);
        const auto tupleArg = predicateBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicateBlock);

        const auto tmp_ctx = QueryCtxT::createChildContext(ctx, predicate_builder, tupleArg);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu params", tmp_ctx.params.size());

        mlir::Value predicateResult = nullptr;
        if (qual && qual->length > 0) {
            if (!qual->elements) {
                PGX_WARNING("Qual list has length but no elements array - continuing without filter");
            } else {
                for (int i = 0; i < qual->length; i++) {
                    const ListCell* lc = &qual->elements[i];
                    if (!lc) {
                        PGX_WARNING("Null ListCell at index %d", i);
                        continue;
                    }

                    const auto qualNode = static_cast<Node*>(lfirst(lc));

                    if (!qualNode) {
                        PGX_WARNING("Null qual node at index %d", i);
                        continue;
                    }

                    mlir::Value condValue;
                    condValue = translate_expression(tmp_ctx, reinterpret_cast<Expr*>(qualNode));

                    if (condValue) {
                        if (!condValue.getType().isInteger(1)) {
                            condValue = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), condValue);
                        }

                        if (!predicateResult) {
                            predicateResult = condValue;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first join predicate");
                        } else {
                            predicateResult = predicate_builder.create<mlir::db::AndOp>(
                                predicate_builder.getUnknownLoc(), mlir::ValueRange{predicateResult, condValue});
                            PGX_LOG(AST_TRANSLATE, DEBUG, "ANDed join predicate %d", i);
                        }
                    } else {
                        PGX_WARNING("Failed to translate qual condition at index %d", i);
                    }
                }
            }
        }

        if (!predicateResult) {
            throw std::runtime_error("We parsed that there were predicates, but got nothing out of it!");
        }
        if (!predicateResult.getType().isInteger(1)) {
            predicateResult = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                              predicateResult);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                         mlir::ValueRange{predicateResult});
    }

    TranslationResult result;
    result.op = selectionOp;
    result.columns = input.columns;
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 4] RESULT: %s", result.toString().c_str());
    return result;
}

auto PostgreSQLASTTranslator::Impl::build_scan_columns(List* targetlist,
                                                       const std::vector<pgx_lower::frontend::sql::ColumnInfo>& allColumns,
                                                       const std::string& table_scope) const
    -> std::vector<TranslationResult::ColumnSchema> {
    std::vector<TranslationResult::ColumnSchema> columns;
    const PostgreSQLTypeMapper type_mapper(context_);

    ListCell* lc;
    foreach (lc, targetlist) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle)
            continue;
        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<Var*>(tle->expr);
            if (var->varattno > 0 && var->varattno <= static_cast<int>(allColumns.size())) {
                const auto& colInfo = allColumns[var->varattno - 1];
                const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                            colInfo.nullable);

                columns.push_back({.table_name = table_scope,
                                   .column_name = colInfo.name,
                                   .type_oid = colInfo.type_oid,
                                   .typmod = colInfo.typmod,
                                   .mlir_type = mlirType,
                                   .nullable = colInfo.nullable});
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

    mlir::Value inputValue = input.op->getResult(0);
    if (!inputValue) {
        PGX_ERROR("Input operation has no result");
        return input;
    }

    // When we have join context, we need to handle ALL target entries, not just computed ones
    // This ensures we project only the requested columns, not all input columns
    bool handleAllEntries = (merged_join_child != nullptr);

    auto targetEntries = std::vector<TargetEntry*>();
    auto computedEntries = std::vector<TargetEntry*>();

    for (int i = 0; i < target_list->length; i++) {
        auto* tle = static_cast<TargetEntry*>(lfirst(&target_list->elements[i]));
        if (tle) {
            if (handleAllEntries) {
                targetEntries.push_back(tle);
                if (tle->expr && tle->expr->type != T_Var) {
                    computedEntries.push_back(tle);
                }
            } else {
                targetEntries.push_back(tle);
                if (tle->expr && tle->expr->type != T_Var) {
                    computedEntries.push_back(tle);
                }
            }
        }
    }

    if (!handleAllEntries && computedEntries.empty()) {
        return input;
    }

    auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    // First pass: Translate expressions to get their types. We need a tuple context, so create a temporary MapOp
    auto expressionTypes = std::vector<mlir::Type>();
    auto columnNames = std::vector<std::string>();
    auto expressionOids = std::vector<Oid>();
    {
        auto placeholderAttrs = std::vector<mlir::Attribute>();
        for (auto i = 0; i < computedEntries.size(); i++) {
            auto tempName = std::string("temp_") + std::to_string(i);
            auto attr = columnManager.createDef(COMPUTED_EXPRESSION_SCOPE, tempName);
            attr.getColumn().type = mlir::NoneType::get(&context_);
            placeholderAttrs.push_back(attr);
        }

        auto tempMapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), inputValue,
                                                                 ctx.builder.getArrayAttr(placeholderAttrs));

        {
            auto& tempRegion = tempMapOp.getPredicate();
            auto* tempBlock = &tempRegion.emplaceBlock();
            auto tupleArg = tempBlock->addArgument(mlir::relalg::TupleType::get(&context_), ctx.builder.getUnknownLoc());

            mlir::OpBuilder temp_builder(&context_);
            temp_builder.setInsertionPointToStart(tempBlock);
            auto tmp_ctx = QueryCtxT::createChildContext(ctx, temp_builder, tupleArg);

            for (auto* entry : computedEntries) {
                auto colName = entry->resname ? entry->resname : "col_" + std::to_string(entry->resno);
                if (colName == "?column?")
                    colName = "col_" + std::to_string(entry->resno);

                // TODO: NV: This is bad. This should be using the TranslationResult to find the name. Actually, most of
                // this function
                //           seems kind of bad to me. But oh well, it's working for now I guess.
                if (!entry->resname && ctx.current_stmt.planTree) {
                    const Plan* topPlan = ctx.current_stmt.planTree;
                    const Agg* aggNode = nullptr;

                    if (topPlan->type == T_Agg) {
                        aggNode = reinterpret_cast<const Agg*>(topPlan);
                    } else if (topPlan->type == T_Sort && topPlan->lefttree && topPlan->lefttree->type == T_Agg) {
                        aggNode = reinterpret_cast<const Agg*>(topPlan->lefttree);
                    }

                    if (aggNode && aggNode->plan.targetlist) {
                        ListCell* lc;
                        int idx = 0;
                        foreach (lc, aggNode->plan.targetlist) {
                            idx++;
                            if (idx == entry->resno) {
                                const auto* aggTe = static_cast<const TargetEntry*>(lfirst(lc));
                                if (aggTe->resname) {
                                    colName = aggTe->resname;
                                    PGX_LOG(AST_TRANSLATE, DEBUG,
                                            "MapOp: Using name '%s' from parent Agg's targetlist for expression",
                                            colName.c_str());
                                }
                                break;
                            }
                        }
                    }
                }

                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "MapOp: Creating computed column '%s' from targetentry resno=%d resname='%s'", colName.c_str(),
                        entry->resno, entry->resname ? entry->resname : "<null>");

                if (mlir::Value exprValue = translate_expression(tmp_ctx, entry->expr)) {
                    mlir::Type exprMlirType = exprValue.getType();
                    expressionTypes.push_back(exprMlirType);
                    columnNames.push_back(colName);
                    Oid typeOid = PostgreSQLTypeMapper::map_mlir_type_to_oid(exprMlirType);
                    expressionOids.push_back(typeOid);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "MapOp column '%s': MLIR type mapped to OID=%u",
                            colName.c_str(), typeOid);
                } else {
                    PGX_WARNING("Failed to get expression!!");
                }
            }
        }
        tempMapOp.erase();

        if (expressionTypes.empty()) {
            return input;
        }
    }

    mlir::relalg::MapOp mapOp;
    {
        std::vector<mlir::Attribute> computedColAttrs;
        for (size_t i = 0; i < expressionTypes.size(); i++) {
            auto columnPtr = columnManager.get(COMPUTED_EXPRESSION_SCOPE, columnNames[i]);
            columnPtr->type = expressionTypes[i];
            computedColAttrs.push_back(columnManager.createDef(COMPUTED_EXPRESSION_SCOPE, columnNames[i]));
        }

        mapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), inputValue,
                                                        ctx.builder.getArrayAttr(computedColAttrs));

        // Build computation region
        auto& predicateRegion = mapOp.getPredicate();
        auto* predicateBlock = new mlir::Block;
        predicateRegion.push_back(predicateBlock);
        auto tupleArg = predicateBlock->addArgument(mlir::relalg::TupleType::get(&context_), ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicateBlock);
        auto tmp_ctx = QueryCtxT::createChildContext(ctx, predicate_builder, tupleArg);

        std::vector<mlir::Value> computedValues;
        for (auto* entry : computedEntries) {
            mlir::Value exprValue;
            if (merged_join_child != nullptr) {
                exprValue = translate_expression(tmp_ctx, entry->expr);
            } else {
                exprValue = translate_expression(tmp_ctx, entry->expr);
            }

            if (exprValue) {
                computedValues.push_back(exprValue);
            } else {
                PGX_WARNING("Failed to get expression!!");
            }
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), computedValues);
    }

    std::vector<TranslationResult::ColumnSchema> allColumns = input.columns;
    for (size_t i = 0; i < expressionTypes.size(); i++) {
        allColumns.push_back({.table_name = COMPUTED_EXPRESSION_SCOPE,
                             .column_name = columnNames[i],
                             .type_oid = expressionOids[i],
                             .typmod = -1,
                             .mlir_type = expressionTypes[i],
                             .nullable = true});
    }

    TranslationResult intermediateResult;
    intermediateResult.op = mapOp;

    if (!handleAllEntries) {
        // Build result columns in TARGETLIST ORDER (not input-first order)
        // Include ALL entries (both resjunk and non-resjunk) for downstream operations like Sort
        size_t computedIdx = 0;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Building columns from %zu targetEntries, input has %zu columns, %zu computed",
                targetEntries.size(), input.columns.size(), expressionTypes.size());
        for (size_t i = 0; i < targetEntries.size(); i++) {
            auto* tle = targetEntries[i];

            PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] resno=%d, resjunk=%d, expr type=%d",
                    i, tle->resno, tle->resjunk, tle->expr ? tle->expr->type : -1);

            // Include ALL targetlist entries (don't skip resjunk here - materialize will filter)
            if (tle->expr && tle->expr->type == T_Var) {
                // This is a Var - find it in input.columns by name, not by varattno
                const auto* var = reinterpret_cast<const Var*>(tle->expr);
                PGX_LOG(AST_TRANSLATE, DEBUG, "    Var: varno=%d, varattno=%d", var->varno, var->varattno);

                // Resolve the Var to get table and column name
                std::string tableName, colName;
                bool nullable = false;

                std::optional<int> varnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varnosyn) : std::nullopt;
                std::optional<int> varattnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varattnosyn) : std::nullopt;

                if (auto resolved = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt)) {
                    tableName = resolved->table_name;
                    colName = resolved->column_name;
                    nullable = resolved->nullable;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Resolved via varno_resolution to %s.%s", tableName.c_str(), colName.c_str());
                } else {
                    // Fallback: use PostgreSQL catalog
                    int schema_varno = IS_SPECIAL_VARNO(var->varno) ? var->varnosyn : var->varno;
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
                        intermediateResult.columns.push_back(col);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Column %s.%s not found in input", tableName.c_str(), colName.c_str());
                }
            } else {
                // This is a computed expression - use the computedIdx
                PGX_LOG(AST_TRANSLATE, DEBUG, "    Computed expr, computedIdx=%zu", computedIdx);
                if (computedIdx < expressionTypes.size()) {
                    size_t columnIndex = input.columns.size() + computedIdx;
                    const auto& col = allColumns[columnIndex];
                    PGX_LOG(AST_TRANSLATE, DEBUG, "    Adding computed: %s.%s", col.table_name.c_str(), col.column_name.c_str());
                    intermediateResult.columns.push_back(col);
                    computedIdx++;
                }
            }
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "apply_projection returning intermediateResult with %zu columns (handleAllEntries=false, includes resjunk):", intermediateResult.columns.size());
        for (size_t i = 0; i < intermediateResult.columns.size(); i++) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s", i, intermediateResult.columns[i].table_name.c_str(), intermediateResult.columns[i].column_name.c_str());
        }
        return intermediateResult;
    }

    intermediateResult.columns = allColumns;

    // When handling all entries we need to add a ProjectionOp to select only the columns from the target list
    std::vector<mlir::Attribute> projectedColumnRefs;
    std::vector<TranslationResult::ColumnSchema> projectedColumns;

    size_t computedIdx = 0;
    for (auto* tle : targetEntries) {
        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<const Var*>(tle->expr);

            size_t left_column_count = merged_join_child ? merged_join_child->left_child_column_count : 0;
            size_t right_column_count = merged_join_child ? (merged_join_child->columns.size() - left_column_count) : 0;
            bool inputContainsBothSides = merged_join_child
                                          && (input.columns.size() >= merged_join_child->columns.size());

            size_t columnIndex = SIZE_MAX;
            if (var->varno == OUTER_VAR && merged_join_child && left_column_count > 0) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(left_column_count)) {
                    columnIndex = var->varattno - 1;
                }
            } else if (var->varno == INNER_VAR && merged_join_child && right_column_count > 0) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(right_column_count)) {
                    if (inputContainsBothSides) {
                        columnIndex = left_column_count + (var->varattno - 1);
                    } else {
                        columnIndex = var->varattno - 1;
                    }
                }
            } else if (var->varattno > 0 && var->varattno <= static_cast<int>(input.columns.size())) {
                columnIndex = var->varattno - 1;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving Var (varno=%d, varattno=%d) to input column %zu: %s.%s",
                        var->varno, var->varattno, columnIndex, input.columns[columnIndex].table_name.c_str(),
                        input.columns[columnIndex].column_name.c_str());
            } else {
                throw std::runtime_error("Failed");
            }

            if (columnIndex < intermediateResult.columns.size()) {
                const auto& col = intermediateResult.columns[columnIndex];
                auto colRef = columnManager.createRef(col.table_name, col.column_name);
                projectedColumnRefs.push_back(colRef);
                projectedColumns.push_back(col);
            }
        } else {
            size_t columnIndex = input.columns.size() + computedIdx;
            if (columnIndex < intermediateResult.columns.size()) {
                const auto& col = intermediateResult.columns[columnIndex];
                auto colRef = columnManager.createRef(col.table_name, col.column_name);
                projectedColumnRefs.push_back(colRef);
                projectedColumns.push_back(col);
                computedIdx++;
            }
        }
    }

    // Create ProjectionOp
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    const auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
        ctx.builder.getUnknownLoc(), tupleStreamType,
        mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all), mapOp.getResult(),
        ctx.builder.getArrayAttr(projectedColumnRefs), mlir::IntegerAttr());

    TranslationResult result;
    result.op = projectionOp;
    result.columns = projectedColumns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_translation_result(
    const QueryCtxT& ctx, const TranslationResult& input, const TranslationResult& merged_join_child,
    const List* target_list, const JoinType join_type) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    if (!input.op || !target_list || target_list->length <= 0) {
        PGX_WARNING("No target list");
        return input;
    }

    auto inputValue = input.op->getResult(0);
    if (!inputValue) {
        PGX_ERROR("Input operation has no result");
        return input;
    }

    std::vector<TranslationResult::ColumnSchema> projectedColumns;
    std::vector<mlir::Attribute> columnRefs;
    auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    size_t left_column_count = merged_join_child.left_child_column_count;
    size_t right_column_count = merged_join_child.columns.size() - left_column_count;

    ListCell* lc;
    foreach (lc, target_list) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle)
            continue;

        if (tle->expr && IsA(tle->expr, Var)) {
            const auto* var = reinterpret_cast<Var*>(tle->expr);
            size_t columnIndex = SIZE_MAX;

            if (var->varno == OUTER_VAR) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(left_column_count)) {
                    columnIndex = var->varattno - 1;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: OUTER_VAR varattno=%d maps to position %zu",
                            var->varattno, columnIndex);
                }
            } else if (var->varno == INNER_VAR) {
                const bool is_exists_join = (join_type == JOIN_SEMI || join_type == JOIN_ANTI
                                             || join_type == JOIN_RIGHT_ANTI);
                if (is_exists_join) {
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Projection: Skipping INNER_VAR reference after %s join (right columns not in output)",
                            join_type == JOIN_SEMI ? "SEMI" : (join_type == JOIN_ANTI ? "ANTI" : "RIGHT_ANTI"));
                    continue;
                }

                std::optional<int> varnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varnosyn)
                                                                               : std::nullopt;
                std::optional<int> varattnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varattnosyn)
                                                                                  : std::nullopt;

                if (auto mapping = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt)) {
                    const auto& table_name = mapping->table_name;
                    const auto& col_name = mapping->column_name;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR using varno_resolution -> @%s::@%s",
                            table_name.c_str(), col_name.c_str());

                    for (size_t i = 0; i < input.columns.size(); ++i) {
                        if (input.columns[i].table_name == table_name && input.columns[i].column_name == col_name) {
                            columnIndex = i;
                            break;
                        }
                    }
                }

                if (columnIndex == SIZE_MAX && var->varattno > 0 && var->varattno <= static_cast<int>(right_column_count))
                {
                    columnIndex = left_column_count + (var->varattno - 1);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR varattno=%d maps to position %zu (fallback)",
                            var->varattno, columnIndex);
                }
            } else {
                PGX_WARNING("Unexpected varno %d in join projection", var->varno);
                continue;
            }

            if (columnIndex < input.columns.size()) {
                const auto& col = input.columns[columnIndex];
                projectedColumns.push_back(col);

                auto colRef = columnManager.createRef(col.table_name, col.column_name);
                columnRefs.push_back(colRef);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Projecting column: %s.%s from position %zu", col.table_name.c_str(),
                        col.column_name.c_str(), columnIndex);
            } else {
                PGX_ERROR("Column index %zu out of bounds (have %zu columns)", columnIndex, input.columns.size());
            }
        } else if (tle->expr) {
            PGX_LOG(AST_TRANSLATE, DEBUG,
                    "Non-Var expression in join projection, delegating to apply_projection_from_target_list");
            auto result = apply_projection_from_target_list(ctx, input, target_list, &merged_join_child);
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
            return result;
        }
    }

    const auto columns_identical = [&]() {
        if (projectedColumns.size() != input.columns.size())
            return false;
        for (size_t i = 0; i < projectedColumns.size(); ++i) {
            if (projectedColumns[i].table_name != input.columns[i].table_name
                || projectedColumns[i].column_name != input.columns[i].column_name)
                return false;
        }
        return true;
    };

    if (!projectedColumns.empty() && !columns_identical()) {
        auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
        const auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
            ctx.builder.getUnknownLoc(), tupleStreamType,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all), inputValue,
            ctx.builder.getArrayAttr(columnRefs), mlir::IntegerAttr());

        TranslationResult result;
        result.op = projectionOp;
        result.columns = projectedColumns;

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created ProjectionOp: projecting %zu columns from %zu input columns",
                projectedColumns.size(), input.columns.size());
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
        return result;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] No projection needed - columns already in correct order");
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", input.toString().c_str());
    return input;
}

auto PostgreSQLASTTranslator::Impl::create_materialize_op(const QueryCtxT& context, const mlir::Value tuple_stream,
                                                          const TranslationResult& translation_result) const
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!translation_result.columns.empty()) {
        auto& columnManager = context.builder.getContext()
                                  ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                                  ->getColumnManager();
        std::vector<mlir::Attribute> columnRefAttrs;
        std::vector<mlir::Attribute> columnNameAttrs;

        const auto* topPlan = context.current_stmt.planTree;
        const auto* targetList = topPlan ? topPlan->targetlist : nullptr;

        for (size_t colIndex = 0; colIndex < translation_result.columns.size(); colIndex++) {
            const auto& column = translation_result.columns[colIndex];
            if (targetList && colIndex < static_cast<size_t>(list_length(targetList))) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(targetList, colIndex));
                if (tle && tle->resjunk) {
                    continue;
                }
            }

            auto outputName = column.column_name;

            if (targetList && colIndex < static_cast<size_t>(list_length(targetList))) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(targetList, colIndex));
                if (tle && tle->resname) {
                    outputName = tle->resname;
                }
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "MaterializeOp column %zu: %s.%s -> output name '%s'", colIndex,
                    column.table_name.c_str(), column.column_name.c_str(), outputName.c_str());

            auto colRef = columnManager.createRef(column.table_name, column.column_name);
            columnRefAttrs.push_back(colRef);

            auto nameAttr = context.builder.getStringAttr(outputName);
            columnNameAttrs.push_back(nameAttr);
        }

        auto columnRefs = context.builder.getArrayAttr(columnRefAttrs);
        auto columnNames = context.builder.getArrayAttr(columnNameAttrs);
        auto tableType = mlir::dsa::TableType::get(&context_);

        auto materializeOp = context.builder.create<mlir::relalg::MaterializeOp>(
            context.builder.getUnknownLoc(), tableType, tuple_stream, columnRefs, columnNames);
        return materializeOp.getResult();
    } else {
        throw std::runtime_error("Should be impossible");
    }
    return mlir::Value();
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
