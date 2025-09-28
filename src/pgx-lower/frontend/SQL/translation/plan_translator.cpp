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

auto PostgreSQLASTTranslator::Impl::process_init_plans(QueryCtxT& ctx, Plan* plan) -> void {
    PGX_IO(AST_TRANSLATE);

    if (!plan->initPlan || list_length(plan->initPlan) == 0) {
        return;
    }

    List* all_subplans = ctx.current_stmt.subplans;
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

        List* setParam = subplan->setParam;
        if (!setParam || list_length(setParam) == 0) {
            PGX_ERROR("InitPlan has no setParam");
            continue;
        }
        const int paramid = list_nth_int(setParam, 0);
        ctx.init_plan_results[paramid] = initplan_result;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Stored InitPlan result for paramid=%d (plan_id=%d, %zu columns)",
                paramid, plan_id, initplan_result.columns.size());
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Processed %d InitPlans, context now has %zu total",
            list_length(plan->initPlan), ctx.init_plan_results.size());
}

auto PostgreSQLASTTranslator::Impl::translate_plan_node(QueryCtxT& ctx, Plan* plan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!plan) {
        PGX_ERROR("Plan node is null");
        throw std::runtime_error("Plan node is null");
    }

    const size_t init_plans_before = ctx.init_plan_results.size();
    process_init_plans(ctx, plan);
    PGX_LOG(AST_TRANSLATE, DEBUG, "After processing InitPlans: context has %zu InitPlans (%zu new)",
            ctx.init_plan_results.size(), ctx.init_plan_results.size() - init_plans_before);

    TranslationResult result;

    switch (plan->type) {
    case T_SeqScan:
    case T_IndexScan:
    case T_IndexOnlyScan:
    case T_BitmapHeapScan: {
            const char* scan_type = (plan->type == T_SeqScan) ? "SeqScan" :
                                    (plan->type == T_IndexScan) ? "IndexScan" :
                                    (plan->type == T_IndexOnlyScan) ? "IndexOnlyScan" : "BitmapHeapScan";

            auto* scan = reinterpret_cast<SeqScan*>(plan);
            result = translate_seq_scan(ctx, scan);

            if (plan->type == T_IndexScan) {
                auto* indexScan = reinterpret_cast<IndexScan*>(plan);
                if (indexScan->indexqual && indexScan->indexqual->length > 0) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "IndexScan has %d indexqual predicates - will be handled by parent NestLoop", indexScan->indexqual->length);
                }
            } else if (plan->type == T_IndexOnlyScan) {
                auto* indexOnlyScan = reinterpret_cast<IndexOnlyScan*>(plan);
                if (indexOnlyScan->indexqual && indexOnlyScan->indexqual->length > 0) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "IndexOnlyScan has %d indexqual predicates - will be handled by parent NestLoop", indexOnlyScan->indexqual->length);
                }
            }

            if (result.op && plan->qual) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "%s has qual, calling apply_selection (context has %zu InitPlans)",
                        scan_type, ctx.init_plan_results.size());
                result = apply_selection_from_qual_with_columns(ctx, result, plan->qual, nullptr, nullptr);
            } else {
                PGX_LOG(AST_TRANSLATE, DEBUG, "%s: no qual (result.op=%p, plan->qual=%p)",
                        scan_type, static_cast<void*>(result.op), static_cast<void*>(plan->qual));
            }

            if (result.op && plan->targetlist) {
                result = apply_projection_from_target_list(ctx, result, plan->targetlist);
            }
        }
        break;
    case T_Agg:
        PGX_LOG(AST_TRANSLATE, DEBUG, "Calling translate_agg");
        result = translate_agg(ctx, reinterpret_cast<Agg*>(plan));
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg returned, result.op=%p", static_cast<void*>(result.op));
        break;
    case T_Sort: result = translate_sort(ctx, reinterpret_cast<Sort*>(plan)); break;
    case T_IncrementalSort:
        PGX_LOG(AST_TRANSLATE, DEBUG, "Treating IncrementalSort as regular Sort"); // TODO: NV: Do I need to deal with
                                                                                   // this?
        result = translate_sort(ctx, reinterpret_cast<Sort*>(plan));
        break;
    case T_Limit: result = translate_limit(ctx, reinterpret_cast<Limit*>(plan)); break;
    case T_Gather: result = translate_gather(ctx, reinterpret_cast<Gather*>(plan)); break;
    case T_MergeJoin: result = translate_merge_join(ctx, reinterpret_cast<MergeJoin*>(plan)); break;
    case T_HashJoin: result = translate_hash_join(ctx, reinterpret_cast<HashJoin*>(plan)); break;
    case T_Hash: result = translate_hash(ctx, reinterpret_cast<Hash*>(plan)); break;
    case T_NestLoop: result = translate_nest_loop(ctx, reinterpret_cast<NestLoop*>(plan)); break;
    case T_Material: result = translate_material(ctx, reinterpret_cast<Material*>(plan)); break;
    case T_SubqueryScan: result = translate_subquery_scan(ctx, reinterpret_cast<SubqueryScan*>(plan)); break;
    case T_CteScan: result = translate_cte_scan(ctx, reinterpret_cast<CteScan*>(plan)); break;
    default: PGX_ERROR("Unsupported plan node type: %d", plan->type); result.op = nullptr;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_plan_node returning result with %zu columns", result.columns.size());
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_seq_scan(QueryCtxT& ctx, SeqScan* seqScan) const -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!seqScan) {
        PGX_ERROR("Invalid SeqScan parameters");
        throw std::runtime_error("Invalid SeqScan parameters");
    }

    auto physicalTableName = std::string();
    auto aliasName = std::string();
    auto tableOid = InvalidOid;

    if (seqScan->scan.scanrelid > 0) {
        physicalTableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);
        aliasName = get_table_alias_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);
        tableOid = get_table_oid_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);

        if (physicalTableName.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", seqScan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", seqScan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    auto tableIdentifier = physicalTableName + TABLE_OID_SEPARATOR + std::to_string(tableOid);

    const auto tableMetaData = std::make_shared<runtime::TableMetaData>();
    tableMetaData->setNumRows(0); // Will be updated from PostgreSQL catalog

    auto tableMetaAttr = mlir::relalg::TableMetaDataAttr::get(&context_, tableMetaData);

    auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto uniqueScope = columnManager.getUniqueScope(aliasName);
    auto columnDefs = std::vector<mlir::NamedAttribute>{};
    auto columnOrder = std::vector<mlir::Attribute>{};
    const auto allColumns = get_all_table_columns_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid);

    if (!allColumns.empty()) {
        int varattno = 1;
        for (const auto& colInfo : allColumns) {
            auto colDef = columnManager.createDef(uniqueScope, colInfo.name);

            PostgreSQLTypeMapper type_mapper(context_);
            const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                        colInfo.nullable);
            colDef.getColumn().type = mlirType;

            columnDefs.push_back(ctx.builder.getNamedAttr(colInfo.name, colDef));
            columnOrder.push_back(ctx.builder.getStringAttr(colInfo.name));

            varattno++;
        }
    } else {
        PGX_ERROR("Could not discover table schema");
        throw std::runtime_error("Could not discover table schema");
    }

    auto columnsAttr = ctx.builder.getDictionaryAttr(columnDefs);
    auto columnOrderAttr = ctx.builder.getArrayAttr(columnOrder);

    const auto baseTableOp = ctx.builder.create<mlir::relalg::BaseTableOp>(
        ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&context_),
        ctx.builder.getStringAttr(tableIdentifier), tableMetaAttr, columnsAttr, columnOrderAttr);

    // Build the output schema from discovered columns
    TranslationResult result;
    result.op = baseTableOp;

    // Only output columns specified in the targetlist
    // The targetlist tells us which columns this node should produce
    if (seqScan->scan.plan.targetlist && seqScan->scan.plan.targetlist->length > 0) {
        ListCell* lc;
        foreach (lc, seqScan->scan.plan.targetlist) {
            auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (!tle || tle->resjunk)
                continue; // Skip junk columns

            if (tle->expr && tle->expr->type == T_Var) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                // Find the column info for this var
                if (var->varattno > 0 && var->varattno <= static_cast<int>(allColumns.size())) {
                    const auto& colInfo = allColumns[var->varattno - 1];
                    PostgreSQLTypeMapper type_mapper(context_);
                    const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                                colInfo.nullable);

                    result.columns.push_back({.table_name = uniqueScope,
                                              .column_name = colInfo.name,
                                              .type_oid = colInfo.type_oid,
                                              .typmod = colInfo.typmod,
                                              .mlir_type = mlirType,
                                              .nullable = colInfo.nullable});
                }
            }
        }
    } else {
        throw std::runtime_error("SeqScan had an empty target list");
    }

    // TODO: NV: Unsure if this is a reasonable solution to my problem...
    if (uniqueScope != aliasName) {
        for (size_t i = 0; i < allColumns.size(); i++) {
            const int varattno = static_cast<int>(i + 1);
            result.varno_resolution[std::make_pair(seqScan->scan.scanrelid, varattno)] =
                std::make_pair(uniqueScope, allColumns[i].name);
        }
    }

    return result;
}

static Aggref* find_first_aggref(Expr* expr) {
    if (!expr) {
        return nullptr;
    }

    if (IsA(expr, Aggref)) {
        return reinterpret_cast<Aggref*>(expr);
    }

    if (IsA(expr, OpExpr)) {
        auto* op_expr = reinterpret_cast<OpExpr*>(expr);
        ListCell* lc;
        foreach(lc, op_expr->args) {
            auto* arg = reinterpret_cast<Expr*>(lfirst(lc));
            Aggref* found = find_first_aggref(arg);
            if (found) {
                return found;
            }
        }
    }

    if (IsA(expr, FuncExpr)) {
        auto* func_expr = reinterpret_cast<FuncExpr*>(expr);
        ListCell* lc;
        foreach(lc, func_expr->args) {
            auto* arg = reinterpret_cast<Expr*>(lfirst(lc));
            Aggref* found = find_first_aggref(arg);
            if (found) {
                return found;
            }
        }
    }

    return nullptr;
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
                            if (existingName.getRootReference().str() == childCol.table_name &&
                                existingName.getLeafReference().str() == childCol.column_name) {
                                alreadyInGroup = true;
                                break;
                            }
                        }
                    }

                    if (!alreadyInGroup) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Adding GROUP BY column from targetlist: %s.%s (ressortgroupref=%d)",
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

            if (te->expr->type == T_Aggref) {
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
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Creating aggregate column definition: scope='%s', column='%s' for aggno=%d",
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
                    auto childCtx = QueryCtxT{ctx.current_stmt, ctx.builder, ctx.current_module, mlir::Value{}, ctx.current_tuple};
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
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Creating aggregate column definition: scope='%s', column='%s' for aggno=%d",
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
            } else if (te->expr->type == T_Var) {
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "First loop: Skipping T_Var at resno=%d (GROUP BY column, handled in second loop)",
                        te->resno);
            } else {
                Aggref* nested_aggref = find_first_aggref(te->expr);

                if (nested_aggref) {
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Found nested Aggref in complex expression at resno=%d, aggno=%d",
                            te->resno, nested_aggref->aggno);

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
                        PGX_LOG(AST_TRANSLATE, DEBUG, "First loop (nested): Added aggregate mapping aggno=%d -> (%s, %s)",
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

                        auto childCtx = QueryCtxT{ctx.current_stmt, ctx.builder, ctx.current_module,
                                                  mlir::Value{}, ctx.current_tuple};
                        childCtx.init_plan_results = ctx.init_plan_results;

                        auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                            childCtx, argTE->expr, childOutput, "nested_agg_expr_" + std::to_string(nested_aggref->aggno),
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
                                                                : type_mapper.map_postgre_sqltype(nested_aggref->aggtype, -1, true);

                        auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                        attrDef.getColumn().type = resultType;

                        aggregateMappings[nested_aggref->aggno] = std::make_pair(aggrScopeName, aggColumnName);
                        PGX_LOG(AST_TRANSLATE, DEBUG, "First loop (nested): Added aggregate mapping aggno=%d -> (%s, %s)",
                                nested_aggref->aggno, aggrScopeName.c_str(), aggColumnName.c_str());

                        auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                            : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                            : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                            : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                                  : mlir::relalg::AggrFunc::count;

                        if (nested_aggref->aggdistinct) {
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing nested %s(DISTINCT) aggregate", funcName.c_str());

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

                        if (attrDef && aggResult) {
                            createdCols.push_back(attrDef);
                            createdValues.push_back(aggResult);
                        }
                    }

                    needs_post_processing.insert(te->resno);
                    post_process_exprs[te->resno] = te->expr;
                    post_process_aggref_map[te->resno] = nested_aggref;

                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Marked resno=%d for post-processing (full expr wraps aggno=%d)",
                            te->resno, nested_aggref->aggno);
                } else {
                    PGX_WARNING("Unexpected non-aggregate, non-Var expression in aggregate targetlist at resno=%d, type=%d",
                                te->resno, te->expr->type);
                }
            }
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

            auto mapOp = ctx.builder.create<mlir::relalg::MapOp>(
                ctx.builder.getUnknownLoc(), aggOp,
                ctx.builder.getArrayAttr({}));

            auto& mapRegion = mapOp.getPredicate();
            auto* mapBlock = new mlir::Block;
            mapRegion.push_back(mapBlock);
            mapBlock->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()),
                                 ctx.builder.getUnknownLoc());

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

                PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing resno=%d, aggno=%d",
                        resno, nested_aggref->aggno);

                auto& [aggr_scope, aggr_colname] = aggregateMappings.at(nested_aggref->aggno);
                auto aggr_colref = columnManager.createRef(aggr_scope, aggr_colname);

                mlir::Value aggr_value = mapBuilder.create<mlir::relalg::GetColumnOp>(
                    ctx.builder.getUnknownLoc(), aggr_colref.getColumn().type,
                    aggr_colref, mapBlock->getArgument(0)).getRes();

                auto postCtx = QueryCtxT{ctx.current_stmt, mapBuilder, ctx.current_module,
                                        mapBlock->getArgument(0), ctx.outer_tuple};
                postCtx.init_plan_results = ctx.init_plan_results;

                mlir::Value post_value = aggr_value;

                if (IsA(full_expr, OpExpr)) {
                    auto* op_expr = reinterpret_cast<OpExpr*>(full_expr);

                    ListCell* arg_lc;
                    std::vector<mlir::Value> arg_values;
                    foreach(arg_lc, op_expr->args) {
                        auto* arg_expr = reinterpret_cast<Expr*>(lfirst(arg_lc));

                        if (IsA(arg_expr, Aggref)) {
                            arg_values.push_back(aggr_value);
                        } else {
                            auto arg_val = translate_expression(postCtx, arg_expr, postProcResult);
                            arg_values.push_back(arg_val);
                        }
                    }

                    if (arg_values.size() == 2) {
                        char* op_name = get_opname(op_expr->opno);

                        if (op_name && strcmp(op_name, "*") == 0) {
                            post_value = mapBuilder.create<mlir::db::MulOp>(
                                ctx.builder.getUnknownLoc(), arg_values[0].getType(),
                                arg_values[0], arg_values[1]);
                        } else if (op_name && strcmp(op_name, "/") == 0) {
                            post_value = mapBuilder.create<mlir::db::DivOp>(
                                ctx.builder.getUnknownLoc(), arg_values[0].getType(),
                                arg_values[0], arg_values[1]);
                        } else if (op_name && strcmp(op_name, "+") == 0) {
                            post_value = mapBuilder.create<mlir::db::AddOp>(
                                ctx.builder.getUnknownLoc(), arg_values[0].getType(),
                                arg_values[0], arg_values[1]);
                        } else if (op_name && strcmp(op_name, "-") == 0) {
                            post_value = mapBuilder.create<mlir::db::SubOp>(
                                ctx.builder.getUnknownLoc(), arg_values[0].getType(),
                                arg_values[0], arg_values[1]);
                        }

                        if (op_name) pfree(op_name);
                    }
                }

                std::string colName = "postproc_" + std::to_string(resno);
                auto colDef = columnManager.createDef(postMapScope, colName);
                colDef.getColumn().type = post_value.getType();

                postMapCols.push_back(colDef);
                mapValues.push_back(post_value);

                PGX_LOG(AST_TRANSLATE, DEBUG, "Created post-processing column: %s.%s",
                        postMapScope.c_str(), colName.c_str());
            }

            mapBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mapValues);

            std::vector<mlir::Attribute> postMapAttrs;
            for (const auto& col : postMapCols) {
                postMapAttrs.push_back(col);
            }
            mapOp.setComputedColsAttr(ctx.builder.getArrayAttr(postMapAttrs));

            finalOutput = mapOp;
            finalScope = postMapScope;

            PGX_LOG(AST_TRANSLATE, DEBUG, "Post-processing MapOp created with %zu columns",
                    postMapCols.size());
        }

        // Build output schema
        TranslationResult result;
        result.op = finalOutput.getDefiningOp();
        result.current_scope = finalScope;

        foreach (lc, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr)
                continue;

            if (te->expr->type == T_Aggref) {
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
                    resultColumnName = mapping.second;  // Use the same column name from the mapping
                    result.varno_resolution[std::make_pair(-2, aggref->aggno)] = mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Added aggregate mapping to TranslationResult: varno=-2, aggno=%d -> (%s, %s)",
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
                PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully pushed aggregate column, result now has %zu columns", result.columns.size());
            } else if (te->expr->type == T_Var) {
                auto* var = reinterpret_cast<Var*>(te->expr);
                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    const auto& childCol = childResult.columns[var->varattno - 1];
                    result.columns.push_back(childCol);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: Adding GROUP BY column '%s' to output (varattno=%d, resname=%s)",
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
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Agg: Adding post-processed column '%s' in scope '%s' (resno=%d)",
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

        if (agg->plan.qual && agg->plan.qual->length > 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing HAVING clause with %d varno_resolution entries",
                    static_cast<int>(result.varno_resolution.size()));
            for (const auto& [key, value] : result.varno_resolution) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "  HAVING: varno=%d, attno=%d -> (%s, %s)",
                        key.first, key.second, value.first.c_str(), value.second.c_str());
            }

            result = apply_selection_from_qual(ctx, result, agg->plan.qual);
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning result with %zu columns, op=%p",
                result.columns.size(), static_cast<void*>(result.op));
        return result;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_agg: Returning childResult with %zu columns",
            childResult.columns.size());
    return childResult;
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

        // Determine sort direction
        auto spec = mlir::relalg::SortSpec::asc;
        if (sort->sortOperators) {
            char* oprname = get_opname(sort->sortOperators[i]);
            if (oprname) {
                spec = (std::string(oprname) == ">" || std::string(oprname) == ">=") ? mlir::relalg::SortSpec::desc
                                                                                     : mlir::relalg::SortSpec::asc;
                pfree(oprname);
            }
        }

        // Find column at position colIdx in Sort's targetlist
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

    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
    const auto sortOp = ctx.builder.create<mlir::relalg::SortOp>(
        ctx.builder.getUnknownLoc(), tupleStreamType, childResult.op->getResult(0), ctx.builder.getArrayAttr(sortSpecs));

    TranslationResult result;
    result.op = sortOp;

    if (sort->plan.targetlist) {
        result.columns.clear();
        ListCell* lc;
        foreach (lc, sort->plan.targetlist) {
            const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (!tle || tle->resjunk)
                continue;

            if (tle->expr && tle->expr->type == T_Var) {
                const auto* var = reinterpret_cast<Var*>(tle->expr);
                if (var->varattno > 0 && var->varattno <= childResult.columns.size()) {
                    result.columns.push_back(childResult.columns[var->varattno - 1]);
                }
            }
        }
    } else {
        result.columns = childResult.columns;
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
        if (node->type == T_Const) {
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
        if (node->type == T_Const) {
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
        limitCount = INT32_MAX; // Use max for "no limit"
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

    // In a full implementation, we would:
    // 1. Create worker coordination logic
    // 2. Handle partial aggregates from workers
    // 3. Implement tuple gathering and merging

    // Gather doesn't change the schema - pass through child's columns
    return childResult; // For now, just pass through the child
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

        // Add tuple argument to the predicate block
        const auto tupleType = mlir::relalg::TupleType::get(&context_);
        const auto tupleArg = predicateBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

        // Set insertion point to predicate block
        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicateBlock);

        auto tmp_ctx = QueryCtxT{ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.outer_tuple};
        tmp_ctx.init_plan_results = ctx.init_plan_results;
            tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
        tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
        tmp_ctx.correlation_params = ctx.correlation_params;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu InitPlans", tmp_ctx.init_plan_results.size());

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

                    // Pass the TranslationResult to translate_expression for proper varno resolution
                    if (mlir::Value condValue = translate_expression(tmp_ctx, reinterpret_cast<Expr*>(qualNode), input)) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully translated HAVING condition %d", i);
                        if (!condValue.getType().isInteger(1)) {
                            condValue = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), condValue);
                        }

                        if (!predicateResult) {
                            predicateResult = condValue;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first HAVING predicate");
                        } else {
                            // TODO: NV: This is hardcoded to only ANDs... it should figure out what the predicate type
                            // is
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

auto PostgreSQLASTTranslator::Impl::apply_selection_from_qual_with_columns(
    const QueryCtxT& ctx, const TranslationResult& input, const List* qual, const TranslationResult* left_child,
    const TranslationResult* right_child) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 3] input: %s", input.toString().c_str());
    if (left_child)
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 3] LEFT input: %s", left_child->toString().c_str());
    if (right_child)
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 3] RIGHT input: %s", right_child->toString().c_str());

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

        auto tmp_ctx = QueryCtxT{ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.outer_tuple};
        tmp_ctx.init_plan_results = ctx.init_plan_results;
        tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
            tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
        tmp_ctx.correlation_params = ctx.correlation_params;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created predicate context with %zu InitPlans", tmp_ctx.init_plan_results.size());

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
                    if (left_child || right_child) {
                        condValue = translate_expression_with_join_context(tmp_ctx, reinterpret_cast<Expr*>(qualNode),
                                                                           left_child, right_child);
                    } else {
                        condValue = translate_expression(tmp_ctx, reinterpret_cast<Expr*>(qualNode), input);
                    }

                    if (condValue) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully translated condition %d (join context: %s)", i,
                                (left_child || right_child) ? "yes" : "no");
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

void PostgreSQLASTTranslator::Impl::translate_join_predicate_to_region(const QueryCtxT& ctx, mlir::Block* predicateBlock,
                                                                       mlir::Value tupleArg, List* joinClauses,
                                                                       const TranslationResult& leftTranslation,
                                                                       const TranslationResult& rightTranslation) {
    PGX_IO(AST_TRANSLATE);

    if (!joinClauses || joinClauses->length == 0) {
        // No join clauses - create constant true
        mlir::OpBuilder predicateBuilder(ctx.builder.getContext());
        predicateBuilder.setInsertionPointToStart(predicateBlock);
        auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
            predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
            predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
        predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
        return;
    }

    mlir::OpBuilder predicateBuilder(ctx.builder.getContext());
    predicateBuilder.setInsertionPointToStart(predicateBlock);
    QueryCtxT predicateCtx(ctx.current_stmt, predicateBuilder, ctx.current_module, tupleArg);

    // Translate each join clause
    std::vector<mlir::Value> conditions;
    ListCell* lc;
    foreach (lc, joinClauses) {
        auto clause = static_cast<Expr*>(lfirst(lc));

        if (clause->type == T_OpExpr) {
            // For join predicates, we need to translate with join context
            mlir::Value conditionValue = translate_expression_with_join_context(predicateCtx, clause, &leftTranslation,
                                                                                &rightTranslation);

            if (conditionValue) {
                conditions.push_back(conditionValue);
            } else {
                PGX_WARNING("Failed to translate join clause");
            }
        } else {
            // Handle other expression types if needed
            mlir::Value conditionValue = translate_expression_with_join_context(predicateCtx, clause, &leftTranslation,
                                                                                &rightTranslation);

            if (conditionValue) {
                conditions.push_back(conditionValue);
            }
        }
    }

    // Combine conditions with AND if multiple
    mlir::Value finalCondition;
    if (conditions.empty()) {
        // No valid conditions translated - return true
        finalCondition = predicateBuilder.create<mlir::arith::ConstantOp>(
            predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
            predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
    } else if (conditions.size() == 1) {
        finalCondition = conditions[0];
    } else {
        // Combine with AND - let the type be inferred
        finalCondition = conditions[0];
        for (size_t i = 1; i < conditions.size(); ++i) {
            // Don't specify return type - let it be inferred based on operands
            finalCondition = predicateBuilder.create<mlir::db::AndOp>(predicateBuilder.getUnknownLoc(),
                                                                      mlir::ValueRange{finalCondition, conditions[i]});
        }
    }

    // Ensure the final condition is non-nullable boolean (i1) for join predicates
    // DeriveTruth converts any type (including nullable) to non-nullable i1
    if (!finalCondition.getType().isInteger(1)) {
        finalCondition = predicateBuilder.create<mlir::db::DeriveTruth>(predicateBuilder.getUnknownLoc(), finalCondition);
    }

    // Return the final condition
    predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{finalCondition});
}


auto PostgreSQLASTTranslator::Impl::apply_projection_from_target_list(
    const QueryCtxT& ctx, const TranslationResult& input, const List* target_list, const TranslationResult* left_child,
    const TranslationResult* right_child) -> TranslationResult {
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
    bool handleAllEntries = (left_child != nullptr || right_child != nullptr);

    auto targetEntries = std::vector<TargetEntry*>();
    auto computedEntries = std::vector<TargetEntry*>();

    for (int i = 0; i < target_list->length; i++) {
        auto* tle = static_cast<TargetEntry*>(lfirst(&target_list->elements[i]));
        if (tle && !tle->resjunk) {
            if (handleAllEntries) {
                targetEntries.push_back(tle);
                if (tle->expr && tle->expr->type != T_Var) {
                    computedEntries.push_back(tle);
                }
            } else if (tle->expr && tle->expr->type != T_Var) {
                computedEntries.push_back(tle);
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
            auto tmp_ctx = QueryCtxT{ctx.current_stmt, temp_builder, ctx.current_module, tupleArg, ctx.current_tuple};
            tmp_ctx.init_plan_results = ctx.init_plan_results;
            tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;

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

                mlir::Value exprValue;
                if (left_child != nullptr || right_child != nullptr) {
                    exprValue = translate_expression_with_join_context(tmp_ctx, entry->expr, left_child, right_child);
                } else {
                    exprValue = translate_expression(tmp_ctx, entry->expr);
                }

                if (exprValue) {
                    expressionTypes.push_back(exprValue.getType());
                    columnNames.push_back(colName);
                    Oid typeOid = exprType(reinterpret_cast<Node*>(entry->expr));
                    expressionOids.push_back(typeOid);
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
        auto tmp_ctx = QueryCtxT{ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.current_tuple};
        tmp_ctx.init_plan_results = ctx.init_plan_results;
        tmp_ctx.subquery_param_mapping = ctx.subquery_param_mapping;

        std::vector<mlir::Value> computedValues;
        for (auto* entry : computedEntries) {
            mlir::Value exprValue;
            if (left_child != nullptr || right_child != nullptr) {
                exprValue = translate_expression_with_join_context(tmp_ctx, entry->expr, left_child, right_child);
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

    // Build intermediate result with MapOp
    TranslationResult intermediateResult;
    intermediateResult.op = mapOp;
    intermediateResult.columns = input.columns;
    for (size_t i = 0; i < expressionTypes.size(); i++) {
        intermediateResult.columns.push_back({.table_name = COMPUTED_EXPRESSION_SCOPE,
                                              .column_name = columnNames[i],
                                              .type_oid = expressionOids[i],
                                              .typmod = -1,
                                              .mlir_type = expressionTypes[i],
                                              .nullable = true});
    }

    if (handleAllEntries) {
        // When handling all entries (join context), we need to add a ProjectionOp
        // to select only the columns from the target list
        std::vector<mlir::Attribute> projectedColumnRefs;
        std::vector<TranslationResult::ColumnSchema> projectedColumns;

        // Build the projection based on target list
        size_t computedIdx = 0;
        for (auto* tle : targetEntries) {
            if (tle->expr && tle->expr->type == T_Var) {
                // This is a simple column reference
                const auto* var = reinterpret_cast<const Var*>(tle->expr);

                // Find the column in intermediateResult
                size_t columnIndex = SIZE_MAX;
                if (var->varno == OUTER_VAR && left_child) {
                    if (var->varattno > 0 && var->varattno <= static_cast<int>(left_child->columns.size())) {
                        columnIndex = var->varattno - 1;
                    }
                } else if (var->varno == INNER_VAR && right_child) {
                    if (var->varattno > 0 && var->varattno <= static_cast<int>(right_child->columns.size())) {
                        columnIndex = left_child->columns.size() + (var->varattno - 1);
                    }
                }

                if (columnIndex < intermediateResult.columns.size()) {
                    const auto& col = intermediateResult.columns[columnIndex];
                    auto colRef = columnManager.createRef(col.table_name, col.column_name);
                    projectedColumnRefs.push_back(colRef);
                    projectedColumns.push_back(col);
                }
            } else {
                // This is a computed expression - it's at the end of intermediateResult.columns
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
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
            mapOp.getResult(), ctx.builder.getArrayAttr(projectedColumnRefs));

        TranslationResult result;
        result.op = projectionOp;
        result.columns = projectedColumns;
        return result;
    } else {
        // Original behavior: return the intermediate result
        return intermediateResult;
    }
}

static List* combine_join_clauses(List* specialized_clauses, List* join_quals, const char* clause_type_name) {
    if (specialized_clauses && join_quals) {
        const auto combined = list_concat(list_copy(specialized_clauses), list_copy(join_quals));
        PGX_LOG(AST_TRANSLATE, DEBUG, "Combined %d %s with %d joinquals = %d total clauses",
                list_length(specialized_clauses), clause_type_name, list_length(join_quals), list_length(combined));
        return combined;
    } else if (specialized_clauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using %d %s only", list_length(specialized_clauses), clause_type_name);
        return specialized_clauses;
    } else if (join_quals) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using %d joinquals only", list_length(join_quals));
        return join_quals;
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "No join clauses (cross join)");
        return nullptr;
    }
}

TranslationResult
PostgreSQLASTTranslator::Impl::create_join_operation(QueryCtxT& ctx, const JoinType join_type, mlir::Value left_value,
                                                     mlir::Value right_value, const TranslationResult& left_translation,
                                                     const TranslationResult& right_translation, List* join_clauses) {
    // Since it's a complex function, all of its functional dependencies are isolated into lambdas. This means I don't
    // have to hop around to understand the function so much.
    PGX_IO(AST_TRANSLATE);

    TranslationResult result;
    const bool isRightJoin = (join_type == JOIN_RIGHT);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] RIGHT input: %s", right_translation.toString().c_str());

    auto translateExpressionFn = [this, isRightJoin](
                                     const QueryCtxT& ctx_p, Expr* expr, const TranslationResult* left_child,
                                     const TranslationResult* right_child,
                                     const std::optional<mlir::Value> outer_tuple_arg = std::nullopt) -> mlir::Value {
        return isRightJoin
                   ? translate_expression_with_join_context(ctx_p, expr, right_child, left_child, outer_tuple_arg)
                   : translate_expression_with_join_context(ctx_p, expr, left_child, right_child, outer_tuple_arg);
    };

    auto translateJoinPredicateToRegion = [translateExpressionFn](
                                              mlir::Block* predicateBlock, const mlir::Value tupleArg,
                                              const TranslationResult& leftTrans, const TranslationResult& rightTrans,
                                              const QueryCtxT& queryCtx, List* clauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Left TranslationResult %s", leftTrans.toString().c_str());
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Right TranslationResult %s", rightTrans.toString().c_str());

        if (!clauses || clauses->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] No join clauses, returning true");
            auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
            predicateBuilder.setInsertionPointToStart(predicateBlock);
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing %d join clauses", clauses->length);

        auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
        predicateBuilder.setInsertionPointToStart(predicateBlock);

        auto predicateCtx = QueryCtxT(queryCtx.current_stmt, predicateBuilder, queryCtx.current_module, tupleArg, mlir::Value());
        predicateCtx.nest_params = queryCtx.nest_params;
        predicateCtx.init_plan_results = queryCtx.init_plan_results;
        predicateCtx.subquery_param_mapping = queryCtx.subquery_param_mapping;
        predicateCtx.correlation_params = queryCtx.correlation_params;
        auto conditions = std::vector<mlir::Value>();
        ListCell* lc;
        int clauseIdx = 0;
        foreach (lc, clauses) {
            const auto clause = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing clause %d of type %d", ++clauseIdx,
                    clause ? clause->type : -1);

            if (auto conditionValue = translateExpressionFn(predicateCtx, clause, &leftTrans, &rightTrans)) {
                conditions.push_back(conditionValue);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Successfully translated clause %d", clauseIdx);
            } else {
                PGX_WARNING("Failed to translate join clause %d", clauseIdx);
            }
        }

        auto finalCondition = mlir::Value();
        if (conditions.empty()) {
            finalCondition = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            finalCondition = conditions[0];
        } else {
            finalCondition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                finalCondition = predicateBuilder.create<mlir::db::AndOp>(
                    predicateBuilder.getUnknownLoc(), mlir::ValueRange{finalCondition, conditions[i]});
            }
        }

        if (!finalCondition.getType().isInteger(1)) {
            finalCondition = predicateBuilder.create<mlir::db::DeriveTruth>(predicateBuilder.getUnknownLoc(),
                                                                            finalCondition);
        }

        predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(),
                                                        mlir::ValueRange{finalCondition});
    };

    auto addPredicateRegion = [&left_translation, &right_translation, join_clauses, translateJoinPredicateToRegion](
                                  mlir::Operation* op, const bool useJoinClauses, const QueryCtxT& queryCtx) {
        mlir::Region* predicateRegion = nullptr;

        if (auto innerJoin = llvm::dyn_cast<mlir::relalg::InnerJoinOp>(op)) {
            predicateRegion = &innerJoin.getPredicate();
        } else if (auto semiJoin = llvm::dyn_cast<mlir::relalg::SemiJoinOp>(op)) {
            predicateRegion = &semiJoin.getPredicate();
        } else if (auto antiJoin = llvm::dyn_cast<mlir::relalg::AntiSemiJoinOp>(op)) {
            predicateRegion = &antiJoin.getPredicate();
        }

        if (!predicateRegion)
            return;

        auto* predicateBlock = new mlir::Block;
        predicateRegion->push_back(predicateBlock);
        const auto tupleType = mlir::relalg::TupleType::get(queryCtx.builder.getContext());
        const auto tupleArg = predicateBlock->addArgument(tupleType, queryCtx.builder.getUnknownLoc());

        if (useJoinClauses && join_clauses) {
            translateJoinPredicateToRegion(predicateBlock, tupleArg, left_translation, right_translation, queryCtx,
                                           join_clauses);
        } else {
            mlir::OpBuilder predicateBuilder(queryCtx.builder.getContext());
            predicateBuilder.setInsertionPointToStart(predicateBlock);
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
        }
    };

    auto buildNullableColumns = [](const auto& columns, const std::string& scope) {
        std::vector<TranslationResult::ColumnSchema> nullableColumns;
        for (const auto& col : columns) {
            auto nullableCol = col;
            nullableCol.table_name = scope;
            nullableCol.nullable = true;
            if (!mlir::isa<mlir::db::NullableType>(col.mlir_type)) {
                nullableCol.mlir_type = mlir::db::NullableType::get(col.mlir_type);
            }
            nullableColumns.push_back(nullableCol);
        }
        return nullableColumns;
    };

    auto createOuterJoinWithNullableMapping =
        [&left_translation, &right_translation, join_clauses, translateJoinPredicateToRegion](
            mlir::Value primaryValue, mlir::Value outerValue, const TranslationResult& outerTranslation,
            const bool isRightJoin, const QueryCtxT& queryCtx) {
            auto& columnManager = queryCtx.builder.getContext()
                                      ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                                      ->getColumnManager();
            auto mappingAttrs = std::vector<mlir::Attribute>();

            const auto outerJoinScope = "oj" + std::to_string(queryCtx.outer_join_counter++);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating outer join with scope: @%s", outerJoinScope.c_str());

            for (const auto& col : outerTranslation.columns) {
                const mlir::Type nullableType = mlir::isa<mlir::db::NullableType>(col.mlir_type)
                                                    ? col.mlir_type
                                                    : mlir::db::NullableType::get(col.mlir_type);

                auto originalColRef = columnManager.createRef(col.table_name, col.column_name);
                const auto fromExistingAttr = queryCtx.builder.getArrayAttr({originalColRef});
                auto nullableColDef = columnManager.createDef(outerJoinScope, col.column_name, fromExistingAttr);
                const auto nullableColPtr = columnManager.get(outerJoinScope, col.column_name);

                nullableColPtr->type = nullableType;
                mappingAttrs.push_back(nullableColDef);
            }

            auto mappingAttr = queryCtx.builder.getArrayAttr(mappingAttrs);

            auto outerJoinOp = queryCtx.builder.create<mlir::relalg::OuterJoinOp>(
                queryCtx.builder.getUnknownLoc(), primaryValue, outerValue, mappingAttr);

            auto& predicateRegion = outerJoinOp.getPredicate();
            auto* predicateBlock = new mlir::Block;
            predicateRegion.push_back(predicateBlock);

            const auto tupleType = mlir::relalg::TupleType::get(queryCtx.builder.getContext());
            const auto tupleArg = predicateBlock->addArgument(tupleType, queryCtx.builder.getUnknownLoc());

            if (isRightJoin) {
                translateJoinPredicateToRegion(predicateBlock, tupleArg, right_translation, left_translation, queryCtx,
                                               join_clauses);
            } else {
                translateJoinPredicateToRegion(predicateBlock, tupleArg, left_translation, right_translation, queryCtx,
                                               join_clauses);
            }

            struct OuterJoinResult {
                mlir::Operation* op;
                std::string scope;
            };

            return OuterJoinResult{outerJoinOp, outerJoinScope};
        };

    const auto buildCorrelatedPredicateRegion = [translateExpressionFn](
                                              mlir::Block* predicateBlock, const mlir::Value outerTupleArg,
                                              const mlir::Value innerTupleArg, List* join_clauses_,
                                              const TranslationResult& leftTrans, const TranslationResult& rightTrans,
                                              const QueryCtxT& queryCtx) {
        auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
        predicateBuilder.setInsertionPointToStart(predicateBlock);
        const auto predicateCtx = QueryCtxT(queryCtx.current_stmt, predicateBuilder, queryCtx.current_module,
                                            innerTupleArg, mlir::Value());
        if (!join_clauses_ || join_clauses_->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] No join clauses, returning true");
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing %d correlation clauses", join_clauses_->length);

        auto conditions = std::vector<mlir::Value>();
        ListCell* lc;
        int clauseIdx = 0;
        foreach (lc, join_clauses_) {
            const auto clause = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing clause %d", ++clauseIdx);

            if (auto conditionValue = translateExpressionFn(predicateCtx, clause, &leftTrans, &rightTrans, outerTupleArg))
            {
                conditions.push_back(conditionValue);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Successfully translated clause %d", clauseIdx);
            } else {
                PGX_WARNING("Failed to translate correlation clause %d", clauseIdx);
            }
        }

        mlir::Value finalCondition;
        if (conditions.empty()) {
            finalCondition = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            finalCondition = conditions[0];
        } else {
            finalCondition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                finalCondition = predicateBuilder.create<mlir::db::AndOp>(
                    predicateBuilder.getUnknownLoc(), mlir::ValueRange{finalCondition, conditions[i]});
            }
        }

        if (!finalCondition.getType().isInteger(1)) {
            finalCondition = predicateBuilder.create<mlir::db::DeriveTruth>(predicateBuilder.getUnknownLoc(),
                                                                            finalCondition);
        }

        predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(),
                                                        mlir::ValueRange{finalCondition});
    };

    const auto buildExistsSubquerySelection = [buildCorrelatedPredicateRegion](
                                                  mlir::Value left_value, mlir::Value right_value, List* join_clauses,
                                                  const bool negate, const TranslationResult& left_trans,
                                                  const TranslationResult& right_trans, const QueryCtxT& query_ctx) {
        auto outer_selection = query_ctx.builder.create<mlir::relalg::SelectionOp>(query_ctx.builder.getUnknownLoc(),
                                                                                    left_value);

        auto& outer_region = outer_selection.getPredicate();
        auto& outer_block = outer_region.emplaceBlock();
        const auto tuple_type = mlir::relalg::TupleType::get(query_ctx.builder.getContext());
        const auto outer_tuple = outer_block.addArgument(tuple_type, query_ctx.builder.getUnknownLoc());

        mlir::OpBuilder outer_builder(&outer_block, outer_block.begin());

        auto inner_selection = outer_builder.create<mlir::relalg::SelectionOp>(outer_builder.getUnknownLoc(),
                                                                                right_value);

        auto& inner_region = inner_selection.getPredicate();
        auto& inner_block = inner_region.emplaceBlock();
        const auto inner_tuple = inner_block.addArgument(tuple_type, outer_builder.getUnknownLoc());

        const auto inner_ctx = QueryCtxT(query_ctx.current_stmt, outer_builder, query_ctx.current_module, inner_tuple, mlir::Value());
        buildCorrelatedPredicateRegion(&inner_block, outer_tuple, inner_tuple, join_clauses, left_trans, right_trans,
                                       inner_ctx);

        auto& col_mgr = query_ctx.builder.getContext()
                            ->template getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                            ->getColumnManager();
        const auto map_scope = col_mgr.getUniqueScope("map");
        auto map_attr = col_mgr.createDef(map_scope, "tmp_attr0");
        map_attr.getColumn().type = outer_builder.getI32Type();

        auto map_op = outer_builder.create<mlir::relalg::MapOp>(outer_builder.getUnknownLoc(),
                                                                inner_selection.getResult(),
                                                                outer_builder.getArrayAttr({map_attr}));

        auto& map_region = map_op.getPredicate();
        auto& map_block = map_region.emplaceBlock();
        map_block.addArgument(tuple_type, outer_builder.getUnknownLoc());

        mlir::OpBuilder map_builder(&map_block, map_block.begin());
        auto const_one = map_builder.create<mlir::db::ConstantOp>(
            map_builder.getUnknownLoc(), map_builder.getI32Type(),
            map_builder.getIntegerAttr(map_builder.getI32Type(), 1));
        map_builder.create<mlir::relalg::ReturnOp>(map_builder.getUnknownLoc(), mlir::ValueRange{const_one});

        auto exists_op = outer_builder.create<mlir::relalg::ExistsOp>(outer_builder.getUnknownLoc(),
                                                                      outer_builder.getI1Type(),
                                                                      map_op.getResult());

        const auto final_value = negate ? outer_builder.create<mlir::db::NotOp>(outer_builder.getUnknownLoc(),
                                                                                 outer_builder.getI1Type(),
                                                                                 exists_op.getResult())
                                                .getResult()
                                        : exists_op.getResult();

        outer_builder.create<mlir::relalg::ReturnOp>(outer_builder.getUnknownLoc(), mlir::ValueRange{final_value});

        return outer_selection.getOperation();
    };

    switch (join_type) {
    case JOIN_INNER: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is an inner join!");
        const auto joinOp = ctx.builder.create<mlir::relalg::InnerJoinOp>(ctx.builder.getUnknownLoc(), left_value,
                                                                          right_value);
        addPredicateRegion(joinOp, true, ctx);
        result.op = joinOp;
        result.columns = left_translation.columns;
        result.columns.insert(result.columns.end(), right_translation.columns.begin(), right_translation.columns.end());
        break;
    }

    case JOIN_SEMI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_SEMI as EXISTS pattern");

        const auto selectionOp = buildExistsSubquerySelection(left_value, right_value, join_clauses, false,
                                                              left_translation, right_translation, ctx);
        result.op = selectionOp;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_ANTI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_ANTI as NOT EXISTS pattern");

        const auto selectionOp = buildExistsSubquerySelection(left_value, right_value, join_clauses, true,
                                                              left_translation, right_translation, ctx);

        result.op = selectionOp;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_LEFT:
    case JOIN_RIGHT: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is a left/right join!");

        const auto [op, scope] = isRightJoin ? createOuterJoinWithNullableMapping(right_value, left_value,
                                                                                  left_translation, true, ctx)
                                             : createOuterJoinWithNullableMapping(left_value, right_value,
                                                                                  right_translation, false, ctx);

        result.op = op;

        const auto& nullableSide = isRightJoin ? left_translation : right_translation;
        const auto& nonNullableSide = isRightJoin ? right_translation : left_translation;

        auto nullableColumns = buildNullableColumns(nullableSide.columns, scope);

        if (isRightJoin) {
            result.columns = nullableColumns;
            result.columns.insert(result.columns.end(), nonNullableSide.columns.begin(), nonNullableSide.columns.end());
        } else {
            result.columns = nonNullableSide.columns;
            result.columns.insert(result.columns.end(), nullableColumns.begin(), nullableColumns.end());
        }

        result.current_scope = scope;
        for (size_t i = 0; i < result.columns.size(); ++i) {
            const auto& col = result.columns[i];
            result.varno_resolution[std::make_pair(-2, i + 1)] = std::make_pair(col.table_name, col.column_name);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Added JOIN mapping to TranslationResult: varno=-2, varattno=%zu -> @%s::@%s",
                    i + 1, col.table_name.c_str(), col.column_name.c_str());
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "%s JOIN created with scope @%s, total columns: %zu",
                isRightJoin ? "RIGHT" : "LEFT", scope.c_str(), result.columns.size());
        break;
    }

    case JOIN_FULL:
        PGX_WARNING("FULL OUTER JOIN not yet fully implemented");
        throw std::runtime_error("FULL OUTER JOIN not yet fully implemented");

    default: PGX_ERROR("Unsupported join type: %d", join_type); throw std::runtime_error("Unsupported join type");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RIGHT input: %s", right_translation.toString().c_str());

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_merge_join(QueryCtxT& ctx, MergeJoin* mergeJoin) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!mergeJoin) {
        PGX_ERROR("Invalid MergeJoin parameters");
        throw std::runtime_error("Invalid MergeJoin parameters");
    }

    auto* leftPlan = mergeJoin->join.plan.lefttree;
    auto* rightPlan = mergeJoin->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("MergeJoin missing left or right child");
        throw std::runtime_error("MergeJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating MergeJoin - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of MergeJoin");
        throw std::runtime_error("Failed to translate left child of MergeJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of MergeJoin");
        throw std::runtime_error("Failed to translate right child of MergeJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    List* combinedClauses = combine_join_clauses(mergeJoin->mergeclauses, mergeJoin->join.joinqual, "mergeclauses");
    TranslationResult result = create_join_operation(ctx, mergeJoin->join.jointype, leftValue, rightValue,
                                                     leftTranslation, rightTranslation, combinedClauses);

    // Join conditions are now handled inside the join predicate region
    // No need to apply them as separate selections

    if (mergeJoin->join.plan.qual) {
        result = apply_selection_from_qual_with_columns(ctx, result, mergeJoin->join.plan.qual, nullptr, nullptr);
    }

    if (mergeJoin->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        result = apply_projection_from_translation_result(ctx, result, leftTranslation, rightTranslation,
                                                          mergeJoin->join.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_hash_join(QueryCtxT& ctx, HashJoin* hashJoin) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!hashJoin) {
        PGX_ERROR("Invalid HashJoin parameters");
        throw std::runtime_error("Invalid HashJoin parameters");
    }

    auto* leftPlan = hashJoin->join.plan.lefttree;
    auto* rightPlan = hashJoin->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("HashJoin missing left or right child");
        throw std::runtime_error("HashJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating HashJoin - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of HashJoin");
        throw std::runtime_error("Failed to translate left child of HashJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of HashJoin");
        throw std::runtime_error("Failed to translate right child of HashJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    List* combinedClauses = combine_join_clauses(hashJoin->hashclauses, hashJoin->join.joinqual, "hashclauses");
    TranslationResult result = create_join_operation(
        ctx, hashJoin->join.jointype, leftValue, rightValue, leftTranslation, rightTranslation, combinedClauses);

    // Join conditions are now handled inside the join predicate region
    // No need to apply them as separate selections

    if (hashJoin->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        result = apply_selection_from_qual_with_columns(ctx, result, hashJoin->join.plan.qual, nullptr, nullptr);
    }

    if (hashJoin->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        result = apply_projection_from_translation_result(ctx, result, leftTranslation, rightTranslation,
                                                          hashJoin->join.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_hash(QueryCtxT& ctx, Hash* hash) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!hash || !hash->plan.lefttree) {
        PGX_ERROR("Invalid Hash parameters");
        throw std::runtime_error("Invalid Hash parameters");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG,
            "Translating Hash node - passing through to child - it just prepares its child for hashing");
    return translate_plan_node(ctx, hash->plan.lefttree);
}

auto PostgreSQLASTTranslator::Impl::translate_nest_loop(QueryCtxT& ctx, NestLoop* nestLoop) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!nestLoop) {
        PGX_ERROR("Invalid NestLoop parameters");
        throw std::runtime_error("Invalid NestLoop parameters");
    }

    auto* leftPlan = nestLoop->join.plan.lefttree;
    auto* rightPlan = nestLoop->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("NestLoop missing left or right child");
        throw std::runtime_error("NestLoop missing children");
    }

    List* effective_join_qual = nestLoop->join.joinqual;

    if (effective_join_qual && effective_join_qual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has joinqual with %d clauses", effective_join_qual->length);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has NO joinqual");
    }

    // Register nest parameters for PARAM resolution during expression translation
    if (nestLoop->nestParams && nestLoop->nestParams->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Parameterized nested loop detected with %d parameters", nestLoop->nestParams->length);

        ListCell* lc;
        foreach (lc, nestLoop->nestParams) {
            auto* nestParam = reinterpret_cast<NestLoopParam*>(lfirst(lc));
            if (nestParam && nestParam->paramval && IsA(nestParam->paramval, Var)) {
                auto* paramVar = reinterpret_cast<Var*>(nestParam->paramval);
                ctx.nest_params[nestParam->paramno] = paramVar;
                PGX_LOG(AST_TRANSLATE, DEBUG, "Registered nest param: paramno=%d -> Var(varno=%d, varattno=%d)",
                        nestParam->paramno, paramVar->varno, paramVar->varattno);
            }
        }
    }

    // Always extract indexqual from inner scan for join-level translation
    // indexqual contains join conditions that need both sides of the join
    if (rightPlan->type == T_IndexScan) {
        auto* indexScan = reinterpret_cast<IndexScan*>(rightPlan);
        if (indexScan->indexqual && indexScan->indexqual->length > 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Extracting %d predicates from IndexScan.indexqual for join-level translation", indexScan->indexqual->length);
            if (effective_join_qual && effective_join_qual->length > 0) {
                effective_join_qual = list_concat(list_copy(effective_join_qual), list_copy(indexScan->indexqual));
            } else {
                effective_join_qual = indexScan->indexqual;
            }
        }
    } else if (rightPlan->type == T_IndexOnlyScan) {
        auto* indexOnlyScan = reinterpret_cast<IndexOnlyScan*>(rightPlan);
        if (indexOnlyScan->indexqual && indexOnlyScan->indexqual->length > 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Extracting %d predicates from IndexOnlyScan.indexqual for join-level translation", indexOnlyScan->indexqual->length);
            if (effective_join_qual && effective_join_qual->length > 0) {
                effective_join_qual = list_concat(list_copy(effective_join_qual), list_copy(indexOnlyScan->indexqual));
            } else {
                effective_join_qual = indexOnlyScan->indexqual;
            }
        }
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NestLoop - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of NestLoop");
        throw std::runtime_error("Failed to translate left child of NestLoop");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of NestLoop");
        throw std::runtime_error("Failed to translate right child of NestLoop");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    TranslationResult result = create_join_operation(ctx, nestLoop->join.jointype, leftValue, rightValue,
                                                     leftTranslation, rightTranslation, effective_join_qual);

    if (nestLoop->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        result = apply_selection_from_qual_with_columns(ctx, result, nestLoop->join.plan.qual, nullptr, nullptr);
    }

    if (nestLoop->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        result = apply_projection_from_translation_result(ctx, result, leftTranslation, rightTranslation,
                                                          nestLoop->join.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_material(QueryCtxT& ctx, Material* material) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!material || !material->plan.lefttree) {
        PGX_ERROR("Invalid Material parameters");
        throw std::runtime_error("Invalid Material parameters");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Material node is a pass-through, translating its child");
    return translate_plan_node(ctx, material->plan.lefttree);
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_translation_result(
    const QueryCtxT& ctx, const TranslationResult& input, const TranslationResult& left_child,
    const TranslationResult& right_child, const List* target_list) -> TranslationResult {
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 5] input: %s", input.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 5] LEFT input: %s", left_child.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 5] RIGHT input: %s", right_child.toString().c_str());

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

    ListCell* lc;
    foreach (lc, target_list) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle || tle->resjunk)
            continue;

        if (tle->expr && tle->expr->type == T_Var) {
            const auto* var = reinterpret_cast<Var*>(tle->expr);
            size_t columnIndex = SIZE_MAX;

            if (var->varno == OUTER_VAR) {
                if (var->varattno > 0 && var->varattno <= static_cast<int>(left_child.columns.size())) {
                    columnIndex = var->varattno - 1; // Convert to 0-based
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: OUTER_VAR varattno=%d maps to position %zu",
                            var->varattno, columnIndex);
                }
            } else if (var->varno == INNER_VAR) {
                if (auto mapping = right_child.resolve_var(var->varnosyn, var->varattno)) {
                    const auto& [table_name, col_name] = *mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR using varno_resolution: varnosyn=%d, varattno=%d -> @%s::@%s",
                            var->varnosyn, var->varattno, table_name.c_str(), col_name.c_str());

                    for (size_t i = 0; i < input.columns.size(); ++i) {
                        if (input.columns[i].table_name == table_name && input.columns[i].column_name == col_name) {
                            columnIndex = i;
                            break;
                        }
                    }
                }

                if (columnIndex == SIZE_MAX && var->varattno > 0 && var->varattno <= static_cast<int>(right_child.columns.size())) {
                    columnIndex = left_child.columns.size() + (var->varattno - 1);
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
            auto result = apply_projection_from_target_list(ctx, input, target_list, &left_child, &right_child);
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
            return result;
        }
    }

    const auto columns_identical = [&]() {
        if (projectedColumns.size() != input.columns.size())
            return false;
        for (size_t i = 0; i < projectedColumns.size(); ++i) {
            if (projectedColumns[i].table_name != input.columns[i].table_name ||
                projectedColumns[i].column_name != input.columns[i].column_name)
                return false;
        }
        return true;
    };

    if (!projectedColumns.empty() && !columns_identical()) {
        auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
        const auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
            ctx.builder.getUnknownLoc(), tupleStreamType,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all), inputValue,
            ctx.builder.getArrayAttr(columnRefs));

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

        size_t colIndex = 0;
        for (const auto& column : translation_result.columns) {
            auto outputName = column.column_name;

            if (targetList && colIndex < static_cast<size_t>(list_length(targetList))) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(targetList, colIndex));
                if (tle && tle->resname && !tle->resjunk) {
                    outputName = tle->resname;
                }
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "MaterializeOp column %zu: %s.%s -> output name '%s'", colIndex,
                    column.table_name.c_str(), column.column_name.c_str(), outputName.c_str());

            auto colRef = columnManager.createRef(column.table_name, column.column_name);
            columnRefAttrs.push_back(colRef);

            auto nameAttr = context.builder.getStringAttr(outputName);
            columnNameAttrs.push_back(nameAttr);

            colIndex++;
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

auto PostgreSQLASTTranslator::Impl::translate_subquery_scan(QueryCtxT& ctx, SubqueryScan* subqueryScan)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    if (!subqueryScan || !subqueryScan->subplan) {
        PGX_ERROR("Invalid SubqueryScan parameters");
        throw std::runtime_error("Invalid SubqueryScan parameters");
    }

    const auto scanrelid = subqueryScan->scan.scanrelid;
    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating SubqueryScan with scanrelid=%d", scanrelid);

    auto result = translate_plan_node(ctx, subqueryScan->subplan);

    if (!result.op) {
        PGX_ERROR("Failed to translate SubqueryScan subplan");
        throw std::runtime_error("Failed to translate SubqueryScan subplan");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan subplan translated with %zu columns", result.columns.size());

    if (scanrelid > 0 && subqueryScan->scan.plan.targetlist) {
        List* targetlist = subqueryScan->scan.plan.targetlist;

        const std::string subquery_alias = get_table_alias_from_rte(&ctx.current_stmt, scanrelid);
        const bool needs_projection = !subquery_alias.empty();

        std::vector<mlir::Attribute> projectionColumns;
        std::vector<TranslationResult::ColumnSchema> newColumns;
        auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

        ListCell* lc;
        int output_attno = 1;

        foreach(lc, targetlist) {
            auto* tle = reinterpret_cast<TargetEntry*>(lfirst(lc));
            if (tle->resjunk) {
                continue;
            }

            if (tle->expr && IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(result.columns.size())) {
                    const auto& col = result.columns[var->varattno - 1];

                    if (needs_projection && tle->resname) {
                        const std::string new_col_name = tle->resname;
                        auto colRef = columnManager.createDef(subquery_alias, new_col_name);
                        colRef.getColumn().type = col.mlir_type;
                        projectionColumns.push_back(colRef);

                        newColumns.push_back({
                            subquery_alias,
                            new_col_name,
                            col.type_oid,
                            col.typmod,
                            col.mlir_type,
                            col.nullable
                        });

                        result.varno_resolution[std::make_pair(scanrelid, output_attno)] =
                            std::make_pair(subquery_alias, new_col_name);

                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "SubqueryScan column aliasing: varno=%d, attno=%d: @%s::@%s -> @%s::@%s",
                                scanrelid, output_attno, col.table_name.c_str(), col.column_name.c_str(),
                                subquery_alias.c_str(), new_col_name.c_str());
                    } else {
                        result.varno_resolution[std::make_pair(scanrelid, output_attno)] =
                            std::make_pair(col.table_name, col.column_name);
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "Mapped SubqueryScan: varno=%d, attno=%d -> subplan column %d (@%s::@%s)",
                                scanrelid, output_attno, var->varattno, col.table_name.c_str(), col.column_name.c_str());
                    }
                }
            }
            output_attno++;
        }

        if (needs_projection && !projectionColumns.empty()) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating projection with %zu aliased columns for subquery '%s'",
                    projectionColumns.size(), subquery_alias.c_str());

            auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
            auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
                ctx.builder.getUnknownLoc(),
                tupleStreamType,
                mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
                result.op->getResult(0),
                ctx.builder.getArrayAttr(projectionColumns)
            );

            result.op = projectionOp.getOperation();
            result.columns = newColumns;
            result.current_scope = subquery_alias;
        }
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_cte_scan(QueryCtxT& ctx, CteScan* cteScan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    if (!cteScan) {
        PGX_ERROR("Invalid CteScan parameters");
        throw std::runtime_error("Invalid CteScan parameters");
    }

    const auto cteParam = cteScan->cteParam;
    const auto ctePlanId = cteScan->ctePlanId;
    const auto scanrelid = cteScan->scan.scanrelid;

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating CteScan with cteParam=%d, ctePlanId=%d, scanrelid=%d",
            cteParam, ctePlanId, scanrelid);

    auto it = ctx.init_plan_results.find(cteParam);
    if (it == ctx.init_plan_results.end()) {
        PGX_ERROR("CTE InitPlan result not found for cteParam=%d", cteParam);
        throw std::runtime_error("CTE InitPlan result not found");
    }

    auto result = it->second;

    if (!result.op) {
        PGX_ERROR("CTE InitPlan has no operation for cteParam=%d", cteParam);
        throw std::runtime_error("CTE InitPlan has no operation");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Found CTE InitPlan result with %zu columns", result.columns.size());

    if (scanrelid > 0 && cteScan->scan.plan.targetlist) {
        List* targetlist = cteScan->scan.plan.targetlist;

        const std::string cte_alias = get_table_alias_from_rte(&ctx.current_stmt, scanrelid);
        const bool needs_projection = !cte_alias.empty();

        std::vector<mlir::Attribute> projectionColumns;
        std::vector<TranslationResult::ColumnSchema> newColumns;
        auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

        ListCell* lc;
        int output_attno = 1;

        foreach(lc, targetlist) {
            auto* tle = reinterpret_cast<TargetEntry*>(lfirst(lc));
            if (tle->resjunk) {
                continue;
            }

            if (tle->expr && IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(result.columns.size())) {
                    const auto& col = result.columns[var->varattno - 1];

                    if (needs_projection && tle->resname) {
                        const std::string new_col_name = tle->resname;
                        auto colRef = columnManager.createDef(cte_alias, new_col_name);
                        colRef.getColumn().type = col.mlir_type;
                        projectionColumns.push_back(colRef);

                        newColumns.push_back({
                            cte_alias,
                            new_col_name,
                            col.type_oid,
                            col.typmod,
                            col.mlir_type,
                            col.nullable
                        });

                        result.varno_resolution[std::make_pair(scanrelid, output_attno)] =
                            std::make_pair(cte_alias, new_col_name);

                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "CteScan column aliasing: varno=%d, attno=%d: @%s::@%s -> @%s::@%s",
                                scanrelid, output_attno, col.table_name.c_str(), col.column_name.c_str(),
                                cte_alias.c_str(), new_col_name.c_str());
                    } else {
                        result.varno_resolution[std::make_pair(scanrelid, output_attno)] =
                            std::make_pair(col.table_name, col.column_name);
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "Mapped CteScan: varno=%d, attno=%d -> CTE column %d (@%s::@%s)",
                                scanrelid, output_attno, var->varattno, col.table_name.c_str(), col.column_name.c_str());
                    }
                }
            }
            output_attno++;
        }

        if (needs_projection && !projectionColumns.empty()) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating projection with %zu aliased columns for CTE '%s'",
                    projectionColumns.size(), cte_alias.c_str());

            auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
            auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
                ctx.builder.getUnknownLoc(),
                tupleStreamType,
                mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
                result.op->getResult(0),
                ctx.builder.getArrayAttr(projectionColumns)
            );

            result.op = projectionOp.getOperation();
            result.columns = newColumns;
            result.current_scope = cte_alias;
        }
    }

    return result;
}

} // namespace postgresql_ast
