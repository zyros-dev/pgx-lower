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

auto PostgreSQLASTTranslator::Impl::translate_plan_node(QueryCtxT& ctx, Plan* plan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!plan) {
        PGX_ERROR("Plan node is null");
        throw std::runtime_error("Plan node is null");
    }

    TranslationResult result;

    switch (plan->type) {
    case T_SeqScan:
        if (plan->type == T_SeqScan) {
            auto* seqScan = reinterpret_cast<SeqScan*>(plan);
            result = translate_seq_scan(ctx, seqScan);

            if (result.op && plan->qual) {
                result = apply_selection_from_qual_with_columns(ctx, result, plan->qual, nullptr, nullptr);
            }

            if (result.op && plan->targetlist) {
                result = apply_projection_from_target_list(ctx, result, plan->targetlist);
            }
        } else {
            PGX_ERROR("Type mismatch for SeqScan");
        }
        break;
    case T_Agg: result = translate_agg(ctx, reinterpret_cast<Agg*>(plan)); break;
    case T_Sort: result = translate_sort(ctx, reinterpret_cast<Sort*>(plan)); break;
    case T_IncrementalSort:
        PGX_LOG(AST_TRANSLATE, DEBUG, "Treating IncrementalSort as regular Sort"); // TODO: NV: Do I need to deal with this?
        result = translate_sort(ctx, reinterpret_cast<Sort*>(plan));
        break;
    case T_Limit: result = translate_limit(ctx, reinterpret_cast<Limit*>(plan)); break;
    case T_Gather: result = translate_gather(ctx, reinterpret_cast<Gather*>(plan)); break;
    case T_MergeJoin: result = translate_merge_join(ctx, reinterpret_cast<MergeJoin*>(plan)); break;
    case T_HashJoin: result = translate_hash_join(ctx, reinterpret_cast<HashJoin*>(plan)); break;
    case T_Hash: result = translate_hash(ctx, reinterpret_cast<Hash*>(plan)); break;
    case T_NestLoop: result = translate_nest_loop(ctx, reinterpret_cast<NestLoop*>(plan)); break;
    case T_Material: result = translate_material(ctx, reinterpret_cast<Material*>(plan)); break;
    default: PGX_ERROR("Unsupported plan node type: %d", plan->type); result.op = nullptr;
    }

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

    auto columnDefs = std::vector<mlir::NamedAttribute>{};
    auto columnOrder = std::vector<mlir::Attribute>{};
    const auto allColumns = get_all_table_columns_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid);

    if (!allColumns.empty()) {
        int varattno = 1;
        for (const auto& colInfo : allColumns) {
            auto colDef = columnManager.createDef(aliasName, colInfo.name);

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
            if (!tle || tle->resjunk) continue; // Skip junk columns

            if (tle->expr && tle->expr->type == T_Var) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                // Find the column info for this var
                if (var->varattno > 0 && var->varattno <= static_cast<int>(allColumns.size())) {
                    const auto& colInfo = allColumns[var->varattno - 1];
                    PostgreSQLTypeMapper type_mapper(context_);
                    const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                                colInfo.nullable);

                    result.columns.push_back({
                        .table_name = aliasName,
                        .column_name = colInfo.name,
                        .type_oid = colInfo.type_oid,
                        .typmod = colInfo.typmod,
                        .mlir_type = mlirType,
                        .nullable = colInfo.nullable
                    });
                }
            }
        }
    } else {
        // Fallback: if no targetlist, output all columns
        for (const auto& colInfo : allColumns) {
            PostgreSQLTypeMapper type_mapper(context_);
            const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                        colInfo.nullable);

            result.columns.push_back({
                .table_name = aliasName,
                .column_name = colInfo.name,
                .type_oid = colInfo.type_oid,
                .typmod = colInfo.typmod,
                .mlir_type = mlirType,
                .nullable = colInfo.nullable
            });
        }
    }

    return result;
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
                PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: GROUP BY column %d: table='%s' name='%s' (from child column at index %d)",
                        i, childCol.table_name.c_str(), childCol.column_name.c_str(), colIdx - 1);
                auto colRef = columnManager.createRef(childCol.table_name, childCol.column_name);
                colRef.getColumn().type = childCol.mlir_type;
                groupByAttrs.push_back(colRef);
            }
        }
    } else {
        // Alternative GROUP BY handling - look for TargetEntries with ressortgroupref
        // These couple of lines caused me so much pain you would not believe it
        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (tle && tle->ressortgroupref > 0 && !tle->resjunk && IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    const auto& childCol = childResult.columns[var->varattno - 1];
                    PGX_LOG(AST_TRANSLATE, DEBUG, "GROUP BY (via ressortgroupref): %s.%s", childCol.table_name.c_str(),
                            childCol.column_name.c_str());

                    auto colRef = columnManager.createRef(childCol.table_name, childCol.column_name);
                    colRef.getColumn().type = childCol.mlir_type;
                    groupByAttrs.push_back(colRef);
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

        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr) continue;

            if (te->expr->type == T_Aggref) {
                auto aggref = reinterpret_cast<Aggref*>(te->expr);
                char* rawFuncName = get_func_name(aggref->aggfnoid);
                if (!rawFuncName) continue;
                std::string funcName(rawFuncName);
                pfree(rawFuncName);

                std::string aggColumnName = te->resname ? te->resname : funcName + "_" + std::to_string(aggref->aggno);
                auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                auto relation = block->getArgument(0);
                mlir::Value aggResult;

                ctx.set_column_mapping(-2, aggref->aggno, aggrScopeName, aggColumnName);

                if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
                    // COUNT(*)
                    attrDef.getColumn().type = ctx.builder.getI64Type();
                    aggResult = aggr_builder.create<mlir::relalg::CountRowsOp>(ctx.builder.getUnknownLoc(),
                                                                               ctx.builder.getI64Type(), relation);
                } else {
                    if (!aggref->args || list_length(aggref->args) == 0) continue;

                    auto argTE = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (!argTE || !argTE->expr) continue;

                    // For aggregate expressions, we need a special context
                    auto childCtx = QueryCtxT{
                        ctx.current_stmt,
                        ctx.builder,
                        ctx.current_module,
                        mlir::Value{},
                        ctx.get_all_column_mappings()
                    };

                    auto [stream, column_ref, column_name, table_name] = translate_expression_for_stream(
                        childCtx, argTE->expr, childOutput, "agg_expr_" + std::to_string(aggref->aggno),
                        childResult.columns);

                    // If a new column was created, it means the child node provided more data than thought
                    if (stream != childOutput) {
                        childOutput = stream.cast<mlir::OpResult>();

                        const auto exprOid = exprType(reinterpret_cast<Node*>(argTE->expr));
                        auto type_mapper = PostgreSQLTypeMapper(*ctx.builder.getContext());
                        auto mlirExprType = type_mapper.map_postgre_sqltype(exprOid, -1, true);

                        childResult.columns.push_back({
                            .table_name = table_name,
                            .column_name = column_name,
                            .type_oid = exprOid,
                            .typmod = -1,
                            .mlir_type = mlirExprType,
                            .nullable = true
                        });
                    }

                    auto columnRef = column_ref;

                    PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                    auto resultType = (funcName == "count") ? ctx.builder.getI64Type()
                                                            : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                    attrDef.getColumn().type = resultType;

                    auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                        : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                        : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                        : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                              : mlir::relalg::AggrFunc::count;

                    // I thought this would be a child node, but turns out its a flag/list... neat!
                    if (aggref->aggdistinct) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Processing %s(DISTINCT) aggregate", funcName.c_str());

                        auto distinctStream = aggr_builder.create<mlir::relalg::ProjectionOp>(
                            ctx.builder.getUnknownLoc(),
                            mlir::relalg::TupleStreamType::get(ctx.builder.getContext()),
                            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(),
                                                              mlir::relalg::SetSemantic::distinct),
                            relation,
                            ctx.builder.getArrayAttr({columnRef})
                        );

                        aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(
                            ctx.builder.getUnknownLoc(), resultType,
                            aggrFuncEnum, distinctStream.getResult(), columnRef
                        );
                    } else {
                        aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(ctx.builder.getUnknownLoc(), resultType,
                                                                                  aggrFuncEnum, relation, columnRef);
                    }
                }

                if (attrDef && aggResult) {
                    createdCols.push_back(attrDef);
                    createdValues.push_back(aggResult);
                }
            }
        }

        aggr_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), createdValues);

        auto aggOp = ctx.builder.create<mlir::relalg::AggregationOp>(ctx.builder.getUnknownLoc(), tupleStreamType,
                                                                     childOutput, ctx.builder.getArrayAttr(groupByAttrs),
                                                                     ctx.builder.getArrayAttr(createdCols));
        aggOp.getAggrFunc().push_back(block);

        // Build output schema
        TranslationResult result;
        result.op = aggOp;

        foreach (lc, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr) continue;

            if (te->expr->type == T_Aggref) {
                auto* aggref = reinterpret_cast<Aggref*>(te->expr);
                PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                auto resultType = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)
                                ? ctx.builder.getI64Type()
                                : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                result.columns.push_back({
                    .table_name = aggrScopeName,
                    .column_name = te->resname ? te->resname : "agg_" + std::to_string(te->resno),
                    .type_oid = aggref->aggtype,
                    .typmod = -1,
                    .mlir_type = resultType,
                    .nullable = true
                });
            } else if (te->expr->type == T_Var) {
                auto* var = reinterpret_cast<Var*>(te->expr);
                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    auto childCol = childResult.columns[var->varattno - 1];
                    auto originalName = childCol.column_name;
                    if (te->resname && childCol.column_name != te->resname) {
                        childCol.column_name = te->resname;
                        PGX_WARNING("Agg: Renaming column in targetlist from '%s' to '%s' (varattno=%d)",
                                originalName.c_str(), te->resname, var->varattno);
                    } else {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Agg: Keeping original column name '%s' in targetlist (varattno=%d)",
                                originalName.c_str(), var->varattno);
                    }
                    result.columns.push_back(childCol);
                }
            }
        }

        if (agg->plan.qual && agg->plan.qual->length > 0) {
            result = apply_selection_from_qual(ctx, result, agg->plan.qual);
        }

        return result;
    }

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
        if (colIdx <= 0 || colIdx >= MAX_COLUMN_INDEX) continue;

        // Determine sort direction
        auto spec = mlir::relalg::SortSpec::asc;
        if (sort->sortOperators) {
            char* oprname = get_opname(sort->sortOperators[i]);
            if (oprname) {
                spec = (std::string(oprname) == ">" || std::string(oprname) == ">=")
                     ? mlir::relalg::SortSpec::desc : mlir::relalg::SortSpec::asc;
                pfree(oprname);
            }
        }

        // Find column at position colIdx in Sort's targetlist
        ListCell* lc;
        int idx = 0;
        foreach (lc, sort->plan.targetlist) {
            if (++idx != colIdx) continue;

            const TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (IsA(tle->expr, Var)) {
                const Var* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= childResult.columns.size()) {
                    const auto& column = childResult.columns[var->varattno - 1];
                    sortSpecs.push_back(
                        mlir::relalg::SortSpecificationAttr::get(
                            ctx.builder.getContext(),
                            columnManager.createRef(column.table_name, column.column_name),
                            spec));
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
        ctx.builder.getUnknownLoc(), tupleStreamType,
        childResult.op->getResult(0), ctx.builder.getArrayAttr(sortSpecs));

    TranslationResult result;
    result.op = sortOp;

    if (sort->plan.targetlist) {
        result.columns.clear();
        ListCell* lc;
        foreach (lc, sort->plan.targetlist) {
            const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (!tle || tle->resjunk) continue;

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
    return childResult;  // For now, just pass through the child
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

        // Create new context preserving column mappings
        const auto tmp_ctx = QueryCtxT{
            ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.get_all_column_mappings()
        };

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
    const QueryCtxT& ctx, const TranslationResult& input, const List* qual,
    const TranslationResult* left_child, const TranslationResult* right_child) -> TranslationResult {

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

    {
        auto& predicateRegion = selectionOp.getPredicate();
        auto* predicateBlock = new mlir::Block;
        predicateRegion.push_back(predicateBlock);

        const auto tupleType = mlir::relalg::TupleType::get(&context_);
        const auto tupleArg = predicateBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

        mlir::OpBuilder predicate_builder(&context_);
        predicate_builder.setInsertionPointToStart(predicateBlock);

        const auto tmp_ctx = QueryCtxT{
            ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, {}
        };

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

                    if (mlir::Value condValue = translate_expression_with_join_context(
                            tmp_ctx, reinterpret_cast<Expr*>(qualNode), left_child, right_child)) {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Successfully translated condition %d (join context: %s)",
                                i, (left_child || right_child) ? "yes" : "no");
                        if (!condValue.getType().isInteger(1)) {
                            condValue = predicate_builder.create<mlir::db::DeriveTruth>(
                                predicate_builder.getUnknownLoc(), condValue);
                        }

                        if (!predicateResult) {
                            predicateResult = condValue;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Set first join predicate");
                        } else {
                            predicateResult = predicate_builder.create<mlir::db::AndOp>(
                                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                                mlir::ValueRange{predicateResult, condValue});
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
    return result;
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_target_list(
    const QueryCtxT& ctx, const TranslationResult& input, const List* target_list, const TranslationResult* left_child,
    const TranslationResult* right_child) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!input.op || !target_list || target_list->length <= 0 || !target_list->elements) {
        return input;
    }

    auto inputValue = input.op->getResult(0);
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
            auto tmp_ctx = QueryCtxT{ctx.current_stmt, temp_builder, ctx.current_module, tupleArg, ctx.get_all_column_mappings()};

            for (auto* entry : computedEntries) {
                auto colName = entry->resname ? entry->resname : "col_" + std::to_string(entry->resno);
                if (colName == "?column?")
                    colName = "col_" + std::to_string(entry->resno);

                // TODO: NV: This is bad. This should be using the TranslationResult to find the name. Actually, most of this function
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
                                    PGX_LOG(AST_TRANSLATE, DEBUG, "MapOp: Using name '%s' from parent Agg's targetlist for expression",
                                            colName.c_str());
                                }
                                break;
                            }
                        }
                    }
                }

                PGX_LOG(AST_TRANSLATE, DEBUG, "MapOp: Creating computed column '%s' from targetentry resno=%d resname='%s'",
                        colName.c_str(), entry->resno, entry->resname ? entry->resname : "<null>");

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
        auto tmp_ctx = QueryCtxT{ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.get_all_column_mappings()};

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
        intermediateResult.columns.push_back({
            .table_name = COMPUTED_EXPRESSION_SCOPE,
            .column_name = columnNames[i],
            .type_oid = expressionOids[i],
            .typmod = -1,
            .mlir_type = expressionTypes[i],
            .nullable = true
        });
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
            ctx.builder.getUnknownLoc(),
            tupleStreamType,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
            mapOp.getResult(),
            ctx.builder.getArrayAttr(projectedColumnRefs)
        );

        TranslationResult result;
        result.op = projectionOp;
        result.columns = projectedColumns;
        return result;
    } else {
        // Original behavior: return the intermediate result
        return intermediateResult;
    }
}

static auto create_outer_join_with_nullable_mapping(
    QueryCtxT& ctx,
    mlir::Value primaryValue,
    mlir::Value outerValue,
    const TranslationResult& outerTranslation,
    bool isRightJoin = false
) -> mlir::Operation* {
    PGX_IO(AST_TRANSLATE);

    auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    std::vector<mlir::Attribute> mappingAttrs;

    for (const auto& col : outerTranslation.columns) {
        mlir::Type nullableType;
        if (col.mlir_type.isa<mlir::db::NullableType>()) {
            nullableType = col.mlir_type;
        } else {
            nullableType = mlir::db::NullableType::get(col.mlir_type);
        }

        auto nullableColName = col.column_name + "_nullable";
        auto originalColRef = columnManager.createRef(col.table_name, col.column_name);

        auto fromExistingAttr = ctx.builder.getArrayAttr({originalColRef});
        auto nullableColDef = columnManager.createDef(col.table_name, nullableColName, fromExistingAttr);

        auto nullableColPtr = columnManager.get(col.table_name, nullableColName);
        nullableColPtr->type = nullableType;

        mappingAttrs.push_back(nullableColDef);
    }

    auto mappingAttr = ctx.builder.getArrayAttr(mappingAttrs);

    auto outerJoinOp = ctx.builder.create<mlir::relalg::OuterJoinOp>(
        ctx.builder.getUnknownLoc(),
        primaryValue,
        outerValue,
        mappingAttr
    );

    // TODO: I dislike this in terms of correctness
    // OuterJoinOp needs a predicate region (empty for now - conditions applied separately)
    auto& predicateRegion = outerJoinOp.getPredicate();
    auto* predicateBlock = new mlir::Block;
    predicateRegion.push_back(predicateBlock);

    auto tupleType = mlir::relalg::TupleType::get(ctx.builder.getContext());
    predicateBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

    mlir::OpBuilder predicateBuilder(ctx.builder.getContext());
    predicateBuilder.setInsertionPointToStart(predicateBlock);
    auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
        predicateBuilder.getUnknownLoc(),
        predicateBuilder.getI1Type(),
        predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1)
    );
    predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});

    return outerJoinOp;
}

static auto create_join_operation(
    QueryCtxT& ctx,
    JoinType jointype,
    mlir::Value leftValue,
    mlir::Value rightValue,
    const TranslationResult& leftTranslation,
    const TranslationResult& rightTranslation
) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    TranslationResult result;

    const char* joinTypeName = nullptr;
    switch (jointype) {
        case JOIN_INNER: joinTypeName = "INNER"; break;
        case JOIN_LEFT: joinTypeName = "LEFT"; break;
        case JOIN_RIGHT: joinTypeName = "RIGHT"; break;
        case JOIN_FULL: joinTypeName = "FULL"; break;
        case JOIN_SEMI: joinTypeName = "SEMI"; break;
        case JOIN_ANTI: joinTypeName = "ANTI"; break;
        default: joinTypeName = "UNKNOWN"; break;
    }
    PGX_LOG(AST_TRANSLATE, DEBUG, "Creating join operation of type: %s", joinTypeName);

    switch (jointype) {
        case JOIN_INNER: {
            const auto crossProductOp = ctx.builder.create<mlir::relalg::CrossProductOp>(
                ctx.builder.getUnknownLoc(),
                leftValue,
                rightValue
            );
            result.op = crossProductOp;
            result.columns = leftTranslation.columns;
            result.columns.insert(result.columns.end(), rightTranslation.columns.begin(), rightTranslation.columns.end());
            break;
        }

        case JOIN_LEFT: {
            // LEFT OUTER JOIN - right side becomes nullable
            auto outerJoinOp = create_outer_join_with_nullable_mapping(ctx, leftValue, rightValue, rightTranslation, false);
            result.op = outerJoinOp;
            result.columns = leftTranslation.columns;

            for (auto col : rightTranslation.columns) {
                col.column_name = col.column_name + "_nullable";
                col.nullable = true;
                if (!col.mlir_type.isa<mlir::db::NullableType>()) {
                    col.mlir_type = mlir::db::NullableType::get(col.mlir_type);
                }
                result.columns.push_back(col);
            }
            break;
        }

        case JOIN_RIGHT: {
            // RIGHT OUTER JOIN - swap inputs and make left side nullable
            auto outerJoinOp = create_outer_join_with_nullable_mapping(ctx, rightValue, leftValue, leftTranslation, true);
            result.op = outerJoinOp;

            for (auto col : leftTranslation.columns) {
                col.column_name = col.column_name + "_nullable";
                col.nullable = true;
                if (!col.mlir_type.isa<mlir::db::NullableType>()) {
                    col.mlir_type = mlir::db::NullableType::get(col.mlir_type);
                }
                result.columns.push_back(col);
            }
            result.columns.insert(result.columns.end(), rightTranslation.columns.begin(), rightTranslation.columns.end());
            break;
        }

        case JOIN_FULL: {
            // FULL OUTER JOIN - both sides become nullable
            PGX_WARNING("FULL OUTER JOIN not yet fully implemented - using CrossProductOp as placeholder");
            // TODO: Implement FullOuterJoinOp with proper nullable mapping for both sides
            const auto crossProductOp = ctx.builder.create<mlir::relalg::CrossProductOp>(
                ctx.builder.getUnknownLoc(),
                leftValue,
                rightValue
            );
            result.op = crossProductOp;

            for (auto col : leftTranslation.columns) {
                col.nullable = true;
                col.mlir_type = mlir::db::NullableType::get(col.mlir_type);
                result.columns.push_back(col);
            }
            for (auto col : rightTranslation.columns) {
                col.nullable = true;
                col.mlir_type = mlir::db::NullableType::get(col.mlir_type);
                result.columns.push_back(col);
            }
            break;
        }

        case JOIN_SEMI: {
            // SEMI JOIN - only return left columns
            auto semiJoinOp = ctx.builder.create<mlir::relalg::SemiJoinOp>(
                ctx.builder.getUnknownLoc(),
                leftValue,
                rightValue
            );

            auto& semiPredicateRegion = semiJoinOp.getPredicate();
            auto* semiPredicateBlock = new mlir::Block;
            semiPredicateRegion.push_back(semiPredicateBlock);
            auto semiTupleType = mlir::relalg::TupleType::get(ctx.builder.getContext());
            semiPredicateBlock->addArgument(semiTupleType, ctx.builder.getUnknownLoc());

            mlir::OpBuilder semiPredicateBuilder(ctx.builder.getContext());
            semiPredicateBuilder.setInsertionPointToStart(semiPredicateBlock);
            auto semiTrueVal = semiPredicateBuilder.create<mlir::arith::ConstantOp>(
                semiPredicateBuilder.getUnknownLoc(),
                semiPredicateBuilder.getI1Type(),
                semiPredicateBuilder.getIntegerAttr(semiPredicateBuilder.getI1Type(), 1)
            );
            semiPredicateBuilder.create<mlir::relalg::ReturnOp>(semiPredicateBuilder.getUnknownLoc(), mlir::ValueRange{semiTrueVal});

            result.op = semiJoinOp;
            result.columns = leftTranslation.columns; // Only left columns
            break;
        }

        case JOIN_ANTI: {
            // ANTI SEMI JOIN - only return left columns
            auto antiSemiJoinOp = ctx.builder.create<mlir::relalg::AntiSemiJoinOp>(
                ctx.builder.getUnknownLoc(),
                leftValue,
                rightValue
            );

            // Add predicate region
            auto& antiPredicateRegion = antiSemiJoinOp.getPredicate();
            auto* antiPredicateBlock = new mlir::Block;
            antiPredicateRegion.push_back(antiPredicateBlock);
            auto antiTupleType = mlir::relalg::TupleType::get(ctx.builder.getContext());
            antiPredicateBlock->addArgument(antiTupleType, ctx.builder.getUnknownLoc());

            mlir::OpBuilder antiPredicateBuilder(ctx.builder.getContext());
            antiPredicateBuilder.setInsertionPointToStart(antiPredicateBlock);
            auto antiTrueVal = antiPredicateBuilder.create<mlir::arith::ConstantOp>(
                antiPredicateBuilder.getUnknownLoc(),
                antiPredicateBuilder.getI1Type(),
                antiPredicateBuilder.getIntegerAttr(antiPredicateBuilder.getI1Type(), 1)
            );
            antiPredicateBuilder.create<mlir::relalg::ReturnOp>(antiPredicateBuilder.getUnknownLoc(), mlir::ValueRange{antiTrueVal});

            result.op = antiSemiJoinOp;
            result.columns = leftTranslation.columns;
            break;
        }

        default:
            PGX_ERROR("Unsupported join type: %d", jointype);
            throw std::runtime_error("Unsupported join type");
    }

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

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating MergeJoin - left child type: %d, right child type: %d",
            leftPlan->type, rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto [leftOp, leftColumns] = leftTranslation;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of MergeJoin");
        throw std::runtime_error("Failed to translate left child of MergeJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto [rightOp, rightColumns] = rightTranslation;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of MergeJoin");
        throw std::runtime_error("Failed to translate right child of MergeJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    TranslationResult result = create_join_operation(
        ctx,
        mergeJoin->join.jointype,
        leftValue,
        rightValue,
        leftTranslation,
        rightTranslation
    );

    if (mergeJoin->mergeclauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying merge join conditions from mergeclauses");
        result = apply_selection_from_qual_with_columns(ctx, result, mergeJoin->mergeclauses,
                                                       &leftTranslation, &rightTranslation);
    } else if (mergeJoin->join.joinqual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying merge join conditions from joinqual");
        result = apply_selection_from_qual_with_columns(ctx, result, mergeJoin->join.joinqual,
                                                       &leftTranslation, &rightTranslation);
    }

    if (mergeJoin->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        result = apply_selection_from_qual_with_columns(ctx, result, mergeJoin->join.plan.qual,
                                                       &leftTranslation, &rightTranslation);
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

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating HashJoin - left child type: %d, right child type: %d",
            leftPlan->type, rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto [leftOp, leftColumns] = leftTranslation;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of HashJoin");
        throw std::runtime_error("Failed to translate left child of HashJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto [rightOp, rightColumns] = rightTranslation;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of HashJoin");
        throw std::runtime_error("Failed to translate right child of HashJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    TranslationResult result = create_join_operation(
        ctx,
        hashJoin->join.jointype,
        leftValue,
        rightValue,
        leftTranslation,
        rightTranslation
    );

    if (hashJoin->hashclauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying hash join conditions from hashclauses");
        result = apply_selection_from_qual_with_columns(ctx, result, hashJoin->hashclauses,
                                                       &leftTranslation, &rightTranslation);
    } else if (hashJoin->join.joinqual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying hash join conditions from joinqual");
        result = apply_selection_from_qual_with_columns(ctx, result, hashJoin->join.joinqual,
                                                       &leftTranslation, &rightTranslation);
    }

    if (hashJoin->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        result = apply_selection_from_qual_with_columns(ctx, result, hashJoin->join.plan.qual,
                                                       &leftTranslation, &rightTranslation);
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


    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating Hash node - passing through to child - it just prepares its child for hashing");
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

    if (nestLoop->nestParams && nestLoop->nestParams->length > 0) {
        PGX_WARNING("Parameterized nested loops not yet supported - treating as simple nested loop");
        // TODO: Support parameterized nested loops
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NestLoop - left child type: %d, right child type: %d",
            leftPlan->type, rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto [leftOp, leftColumns] = leftTranslation;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of NestLoop");
        throw std::runtime_error("Failed to translate left child of NestLoop");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto [rightOp, rightColumns] = rightTranslation;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of NestLoop");
        throw std::runtime_error("Failed to translate right child of NestLoop");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    TranslationResult result = create_join_operation(
        ctx,
        nestLoop->join.jointype,
        leftValue,
        rightValue,
        leftTranslation,
        rightTranslation
    );

    if (nestLoop->join.joinqual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying nested loop join conditions from joinqual");
        result = apply_selection_from_qual_with_columns(ctx, result, nestLoop->join.joinqual,
                                                       &leftTranslation, &rightTranslation);
    }

    if (nestLoop->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        result = apply_selection_from_qual_with_columns(ctx, result, nestLoop->join.plan.qual,
                                                       &leftTranslation, &rightTranslation);
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
    const QueryCtxT& ctx, const TranslationResult& input,
    const TranslationResult& left_child, const TranslationResult& right_child,
    const List* target_list) -> TranslationResult {

    PGX_IO(AST_TRANSLATE);
    if (!input.op || !target_list || target_list->length <= 0) {
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
        if (!tle || tle->resjunk) continue;

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
                if (var->varattno > 0 && var->varattno <= static_cast<int>(right_child.columns.size())) {
                    columnIndex = left_child.columns.size() + (var->varattno - 1);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Projection: INNER_VAR varattno=%d maps to position %zu",
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

                PGX_LOG(AST_TRANSLATE, DEBUG, "Projecting column: %s.%s from position %zu",
                        col.table_name.c_str(), col.column_name.c_str(), columnIndex);
            } else {
                PGX_ERROR("Column index %zu out of bounds (have %zu columns)",
                         columnIndex, input.columns.size());
            }
        } else if (tle->expr) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Non-Var expression in join projection, delegating to apply_projection_from_target_list");
            return apply_projection_from_target_list(ctx, input, target_list, &left_child, &right_child);
        }
    }

    // If we only have simple column projections, create a ProjectionOp
    if (!projectedColumns.empty() && projectedColumns.size() < input.columns.size()) {
        auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
        const auto projectionOp = ctx.builder.create<mlir::relalg::ProjectionOp>(
            ctx.builder.getUnknownLoc(),
            tupleStreamType,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
            inputValue,
            ctx.builder.getArrayAttr(columnRefs)
        );

        TranslationResult result;
        result.op = projectionOp;
        result.columns = projectedColumns;

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created ProjectionOp: projecting %zu columns from %zu input columns",
                projectedColumns.size(), input.columns.size());
        return result;
    }

    // No projection needed
    return input;
}

auto PostgreSQLASTTranslator::Impl::create_materialize_op(const QueryCtxT& context, const mlir::Value tuple_stream, const TranslationResult& translation_result) const
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!translation_result.columns.empty()) {
        auto& columnManager = context.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        std::vector<mlir::Attribute> columnRefAttrs;
        std::vector<mlir::Attribute> columnNameAttrs;

        const auto* topPlan = context.current_stmt.planTree;
        const auto* targetList = topPlan ? topPlan->targetlist : nullptr;

        size_t colIndex = 0;
        for (const auto& column : translation_result.columns) {
            auto outputName = column.column_name;

            if (targetList && colIndex < (size_t)list_length(targetList)) {
                const auto* tle = static_cast<TargetEntry*>(list_nth(targetList, colIndex));
                if (tle && tle->resname && !tle->resjunk) {
                    outputName = tle->resname;
                }
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "MaterializeOp column %zu: %s.%s -> output name '%s'",
                    colIndex, column.table_name.c_str(), column.column_name.c_str(), outputName.c_str());

            auto colRef = columnManager.createRef(column.table_name, column.column_name);
            columnRefAttrs.push_back(colRef);

            auto nameAttr = context.builder.getStringAttr(outputName);
            columnNameAttrs.push_back(nameAttr);

            colIndex++;
        }

        auto columnRefs = context.builder.getArrayAttr(columnRefAttrs);
        auto columnNames = context.builder.getArrayAttr(columnNameAttrs);
        auto tableType = mlir::dsa::TableType::get(&context_);

        auto materializeOp = context.builder.create<mlir::relalg::MaterializeOp>(context.builder.getUnknownLoc(), tableType, tuple_stream,
                                                            columnRefs, columnNames);
        return materializeOp.getResult();
    } else {
        throw std::runtime_error("Should be impossible");
    }
    return mlir::Value();
}

} // namespace postgresql_ast
