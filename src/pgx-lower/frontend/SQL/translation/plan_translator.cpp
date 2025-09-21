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
                result = apply_selection_from_qual(ctx, result, plan->qual);
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
    case T_Limit: result = translate_limit(ctx, reinterpret_cast<Limit*>(plan)); break;
    case T_Gather: result = translate_gather(ctx, reinterpret_cast<Gather*>(plan)); break;
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

    std::string tableName;
    auto tableOid = InvalidOid;

    if (seqScan->scan.scanrelid > 0) {
        tableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);
        tableOid = get_table_oid_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);

        if (tableName.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", seqScan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", seqScan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    std::string tableIdentifier = tableName + TABLE_OID_SEPARATOR + std::to_string(tableOid);

    const auto tableMetaData = std::make_shared<runtime::TableMetaData>();
    tableMetaData->setNumRows(0); // Will be updated from PostgreSQL catalog

    auto tableMetaAttr = mlir::relalg::TableMetaDataAttr::get(&context_, tableMetaData);

    auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto columnDefs = std::vector<mlir::NamedAttribute>{};
    auto columnOrder = std::vector<mlir::Attribute>{};
    const auto allColumns = get_all_table_columns_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid);

    if (!allColumns.empty()) {
        std::string realTableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);

        // Populate column mappings for this table
        int varattno = 1; // column numbering starts at 1
        for (const auto& colInfo : allColumns) {
            // Map (scanrelid, varattno) -> (table_name, column_name)
            ctx.column_mappings[{seqScan->scan.scanrelid, varattno}] = {realTableName, colInfo.name};
            PGX_LOG(AST_TRANSLATE, DEBUG, "Mapped (%d, %d) -> (%s, %s)", seqScan->scan.scanrelid, varattno,
                    realTableName.c_str(), colInfo.name.c_str());

            auto colDef = columnManager.createDef(realTableName, colInfo.name);

            PostgreSQLTypeMapper type_mapper(context_);
            const mlir::Type mlirType = type_mapper.map_postgre_sqltype(colInfo.type_oid, colInfo.typmod,
                                                                        colInfo.nullable);
            colDef.getColumn().type = mlirType;

            columnDefs.push_back(ctx.builder.getNamedAttr(colInfo.name, colDef));
            columnOrder.push_back(ctx.builder.getStringAttr(colInfo.name));

            varattno++;
        }

        tableIdentifier = realTableName + TABLE_OID_SEPARATOR
                          + std::to_string(
                              get_all_table_columns_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid).empty()
                                  ? 0
                                  : static_cast<RangeTblEntry*>(
                                        list_nth(ctx.current_stmt.rtable, seqScan->scan.scanrelid - 1))
                                        ->relid);
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
                        .table_name = tableName,
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
                .table_name = tableName,
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

    // Group by
    if (agg->numCols > 0 && agg->grpColIdx) {
        Plan* scanPlan = agg->plan.lefttree;
        while (scanPlan && scanPlan->type != T_SeqScan)
            scanPlan = scanPlan->lefttree;

        if (scanPlan) {
            auto* seqScan = reinterpret_cast<SeqScan*>(scanPlan);
            auto tableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);

            for (int i = 0; i < agg->numCols; i++) {
                int tlIndex = agg->grpColIdx[i] - 1;
                if (tlIndex >= 0 && tlIndex < list_length(agg->plan.targetlist)) {
                    auto* tle = static_cast<TargetEntry*>(list_nth(agg->plan.targetlist, tlIndex));

                    if (tle && tle->expr && IsA(tle->expr, Var)) {
                        auto* var = reinterpret_cast<Var*>(tle->expr);

                        auto columnName = std::string();
                        if (tle->resorigcol > 0) {
                            columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                     tle->resorigcol);
                        } else if (var->varattnosyn > 0) {
                            columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                     var->varattnosyn);
                        } else if (var->varattno > 0) {
                            columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                     var->varattno);
                        }

                        if (!columnName.empty()) {
                            PGX_LOG(AST_TRANSLATE, DEBUG, "GROUP BY col %d: %s.%s (from targetlist pos %d)", i,
                                    tableName.c_str(), columnName.c_str(), tlIndex);
                            groupByAttrs.push_back(
                                columnManager.createRef(columnManager.get(tableName, columnName).get()));
                        } else {
                            PGX_WARNING("GROUP BY col %d: couldn't get column name from Var", i);
                        }
                    } else {
                        // Fallback: try direct column indexing if targetlist doesn't contain a Var
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "GROUP BY col %d: targetlist entry not a Var, using direct indexing", i);
                        auto columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                      agg->grpColIdx[i]);
                        if (!columnName.empty()) {
                            groupByAttrs.push_back(
                                columnManager.createRef(columnManager.get(tableName, columnName).get()));
                        }
                    }
                } else {
                    // TODO: NV Uncomment this
                    // PGX_WARNING("GROUP BY index %d exceeds targetlist length %d - using direct column indexing",
                    //            agg->grpColIdx[i], list_length(agg->plan.targetlist));
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "GROUP BY col %d: index %d exceeds targetlist (len %d), using as column index", i,
                            agg->grpColIdx[i], list_length(agg->plan.targetlist));
                    auto columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                  agg->grpColIdx[i]);
                    if (!columnName.empty()) {
                        groupByAttrs.push_back(columnManager.createRef(columnManager.get(tableName, columnName).get()));
                    }
                }
            }
        }
    } else {
        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto* tle = static_cast<TargetEntry*>(lfirst(lc));
            if (tle && tle->ressortgroupref > 0 && !tle->resjunk && IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);
                std::string tableName, columnName;

                if (var->varno < 0) {
                    Plan* scanPlan = agg->plan.lefttree;
                    while (scanPlan && scanPlan->type != T_SeqScan)
                        scanPlan = scanPlan->lefttree;
                    if (scanPlan) {
                        auto* seqScan = reinterpret_cast<SeqScan*>(scanPlan);
                        tableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);
                        columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                 var->varattnosyn ? var->varattnosyn : var->varattno);
                    } else
                        continue;
                } else {
                    tableName = get_table_name_from_rte(&ctx.current_stmt, var->varno);
                    columnName = get_column_name_from_schema(&ctx.current_stmt, var->varno, var->varattno);
                }
                groupByAttrs.push_back(columnManager.createRef(columnManager.get(tableName, columnName).get()));
            }
        }
    }

    // Process aggregation
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());

    // Static counter for aggregate scope names (like LingoDB's groupById)
    static size_t aggrId = 0;
    std::string aggrScopeName = "aggr" + std::to_string(aggrId++);

    auto savedSortMappings = pgx_lower::frontend::sql::ColumnMapping();
    for (const auto& [fst, snd] : ctx.column_mappings) {
        if (fst.first == -2) {
            savedSortMappings[fst] = snd;
        }
    }

    if (agg->plan.targetlist && agg->plan.targetlist->length > 0) {
        auto* block = new mlir::Block;
        block->addArgument(tupleStreamType, ctx.builder.getUnknownLoc());
        block->addArgument(mlir::relalg::TupleType::get(ctx.builder.getContext()), ctx.builder.getUnknownLoc());

        mlir::OpBuilder aggr_builder(ctx.builder.getContext());
        aggr_builder.setInsertionPointToStart(block);

        auto createdValues = std::vector<mlir::Value>{};
        auto createdCols = std::vector<mlir::Attribute>{};

        size_t aggIndex = 0;
        ListCell* lc;
        foreach (lc, agg->plan.targetlist) {
            auto te = static_cast<TargetEntry*>(lfirst(lc));
            if (!te || !te->expr) {
                PGX_ERROR("Invalid TargetEntry in aggregate plan");
                throw std::runtime_error("Invalid TargetEntry in aggregate plan");
            }

            if (te->expr->type == T_Aggref) {
                auto aggref = reinterpret_cast<Aggref*>(te->expr);
                char* rawFuncName = get_func_name(aggref->aggfnoid);
                if (!rawFuncName) {
                    PGX_ERROR("Unknown aggregate function OID: %u", aggref->aggfnoid);
                    throw std::runtime_error("Unknown aggregate function OID");
                }
                std::string funcName(rawFuncName);
                pfree(rawFuncName);

                // Use the resname from targetlist if available, otherwise generate a name
                std::string aggColumnName = te->resname ? te->resname : funcName + "_" + std::to_string(aggIndex++);
                auto attrDef = columnManager.createDef(aggrScopeName, aggColumnName.c_str());
                auto relation = block->getArgument(0);
                mlir::Value aggResult;

                if (funcName == "count" && (!aggref->args || list_length(aggref->args) == 0)) {
                    // COUNT(*) - no column reference needed, counts all rows including NULLs
                    attrDef.getColumn().type = ctx.builder.getI64Type();
                    aggResult = aggr_builder.create<mlir::relalg::CountRowsOp>(ctx.builder.getUnknownLoc(),
                                                                               ctx.builder.getI64Type(), relation);
                } else { // COUNT(column), SUM, AVG, MIN, MAX - need column reference
                    if (!aggref->args || list_length(aggref->args) == 0) {
                        PGX_ERROR("Aggregate function %s requires arguments", funcName.c_str());
                        continue;
                    }

                    auto argTE = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (!argTE || !argTE->expr || argTE->expr->type != T_Var) {
                        PGX_ERROR("Aggregate function %s requires column reference", funcName.c_str());
                        continue;
                    }

                    auto colVar = reinterpret_cast<Var*>(argTE->expr);
                    std::string tableName, columnName;
                    if (colVar->varno < 0) {
                        if (agg->plan.lefttree && agg->plan.lefttree->type == T_Sort) {
                            auto mappingKey = std::make_pair(-2, colVar->varattno);
                            auto mappingIt = savedSortMappings.find(mappingKey);
                            if (mappingIt != savedSortMappings.end()) {
                                tableName = mappingIt->second.first;
                                columnName = mappingIt->second.second;
                            } else
                                continue;
                        } else {
                            auto* scanPlan = agg->plan.lefttree;
                            while (scanPlan && scanPlan->type != T_SeqScan)
                                scanPlan = scanPlan->lefttree;
                            if (scanPlan) {
                                auto seqScan = reinterpret_cast<SeqScan*>(scanPlan);
                                tableName = get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid);
                                columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                         colVar->varattno);
                            } else
                                continue;
                        }
                    } else {
                        tableName = get_table_name_from_rte(&ctx.current_stmt, colVar->varno);
                        columnName = get_column_name_from_schema(&ctx.current_stmt, colVar->varno, colVar->varattno);
                    }

                    auto nested = std::vector{mlir::FlatSymbolRefAttr::get(ctx.builder.getContext(), columnName)};
                    auto symbolRef = mlir::SymbolRefAttr::get(ctx.builder.getContext(), tableName, nested);
                    auto columnRef = mlir::relalg::ColumnRefAttr::get(ctx.builder.getContext(), symbolRef,
                                                                      columnManager.get(tableName, columnName));

                    PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                    auto resultType = (funcName == "count") ? ctx.builder.getI64Type()
                                                            : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                    attrDef.getColumn().type = resultType;

                    auto aggrFuncEnum = (funcName == "sum")   ? mlir::relalg::AggrFunc::sum
                                        : (funcName == "avg") ? mlir::relalg::AggrFunc::avg
                                        : (funcName == "min") ? mlir::relalg::AggrFunc::min
                                        : (funcName == "max") ? mlir::relalg::AggrFunc::max
                                                              : mlir::relalg::AggrFunc::count;

                    aggResult = aggr_builder.create<mlir::relalg::AggrFuncOp>(ctx.builder.getUnknownLoc(), resultType,
                                                                              aggrFuncEnum, relation, columnRef);
                }

                if (!attrDef || !aggResult)
                    throw std::runtime_error("Failed to generate aggregate");

                createdCols.push_back(attrDef);
                createdValues.push_back(aggResult);

                ctx.column_mappings[{-2, te->resno}] = {aggrScopeName, aggColumnName.c_str()};
            } else if (te->expr && te->expr->type == T_Var) {
                auto var = reinterpret_cast<Var*>(te->expr);
                if (var->varno < 0) {
                    auto* scanPlan = agg->plan.lefttree;
                    while (scanPlan && scanPlan->type != T_SeqScan)
                        scanPlan = scanPlan->lefttree;
                    if (scanPlan) {
                        auto seqScan = reinterpret_cast<SeqScan*>(scanPlan);

                        // Use resorigcol if available (for GROUP BY columns), otherwise use var->varattno
                        auto columnIdx = (te->resorigcol > 0) ? te->resorigcol : var->varattno;
                        auto columnName = get_column_name_from_schema(&ctx.current_stmt, seqScan->scan.scanrelid,
                                                                      columnIdx);

                        ctx.column_mappings[{-2, te->resno}] = {
                            get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid), columnName};

                        PGX_LOG(AST_TRANSLATE, DEBUG, "Agg output mapping: (-2, %d) -> (%s, %s) using col index %d",
                                te->resno, get_table_name_from_rte(&ctx.current_stmt, seqScan->scan.scanrelid).c_str(),
                                columnName.c_str(), columnIdx);
                    }
                }
            }
        }

        aggr_builder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), createdValues);

        auto aggOp = ctx.builder.create<mlir::relalg::AggregationOp>(ctx.builder.getUnknownLoc(), tupleStreamType,
                                                                     childOutput, ctx.builder.getArrayAttr(groupByAttrs),
                                                                     ctx.builder.getArrayAttr(createdCols));
        aggOp.getAggrFunc().push_back(block);

        // Build output schema from GROUP BY columns and aggregates
        TranslationResult result;
        result.op = aggOp;

        // Build output schema based on targetlist
        ListCell* lc2;
        foreach (lc2, agg->plan.targetlist) {
            auto* te = static_cast<TargetEntry*>(lfirst(lc2));
            if (!te || !te->expr) continue;

            if (te->expr->type == T_Aggref) {
                auto* aggref = reinterpret_cast<Aggref*>(te->expr);
                PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
                auto resultType = (aggref->aggfnoid == 2803 || aggref->aggfnoid == 2147)  // COUNT OIDs
                                ? ctx.builder.getI64Type()
                                : type_mapper.map_postgre_sqltype(aggref->aggtype, -1, true);

                std::string columnName = te->resname ? te->resname : "agg_" + std::to_string(te->resno);
                result.columns.push_back({
                    .table_name = aggrScopeName,
                    .column_name = columnName,
                    .type_oid = aggref->aggtype,
                    .typmod = -1,
                    .mlir_type = resultType,
                    .nullable = true
                });
            } else if (te->expr->type == T_Var) {
                auto* var = reinterpret_cast<Var*>(te->expr);
                if (var->varattno > 0 && var->varattno <= static_cast<int>(childResult.columns.size())) {
                    auto childCol = childResult.columns[var->varattno - 1];
                    if (te->resname) {
                        childCol.column_name = te->resname;
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
    result.columns = childResult.columns;
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_limit(QueryCtxT& ctx, const Limit* limit) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!limit) {
        PGX_ERROR("Invalid Limit parameters");
        return TranslationResult{};  // Return empty result
    }

    // Translate child plan - single code path for tests and production
    TranslationResult childResult;

    if (Plan* leftTree = limit->plan.lefttree) {
        childResult = translate_plan_node(ctx, leftTree);
        if (!childResult.op) {
            PGX_ERROR("Failed to translate Limit child plan");
            return childResult;  // Return empty result
        }
    } else {
        PGX_WARNING("Limit node has no child plan");
        // For unit tests, return empty result and handle gracefully
        return TranslationResult{};
    }

    auto childOutput = childResult.op->getResult(0);
    if (!childOutput) {
        PGX_ERROR("Child operation has no result");
        return childResult;
    }

    // Extract actual limit count and offset from the plan
    int64_t limitCount = DEFAULT_LIMIT_COUNT; // Default for unit tests
    int64_t limitOffset = 0;

    Node* limitOffsetNode = limit->limitOffset;

    // In unit tests, limitCountNode might be a mock Const structure
    // In production, it's a real PostgreSQL Node
    // We can safely check the structure and extract values
    if (Node* limitCountNode = limit->limitCount) {
        // Check if this looks like a Const node
        Node* node = limitCountNode;
        if (node->type == T_Const) {
            const Const* constNode = reinterpret_cast<Const*>(node);

            // For unit tests, constvalue directly holds the value
            // For production PostgreSQL, it would be a Datum
            if (!constNode->constisnull) {
                // In unit tests, constvalue is directly the integer value
                // In production, we'd use DatumGetInt32/64
                limitCount = static_cast<int64_t>(constNode->constvalue);
            } else {
            }
        } else if (node->type == T_Param) {
        } else {
            PGX_WARNING("Limit count is not a Const or Param node");
        }
    }

    // Similar handling for offset
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

    // Handle special cases
    if (limitCount == -1) {
        limitCount = INT32_MAX; // Use max for "no limit"
    }

    const auto limitOp = ctx.builder.create<mlir::relalg::LimitOp>(
        ctx.builder.getUnknownLoc(), ctx.builder.getI32IntegerAttr(static_cast<int32_t>(limitCount)), childOutput);

    // Limit doesn't change the schema - pass through child's columns
    TranslationResult result;
    result.op = limitOp;
    result.columns = childResult.columns;  // Inherit child's schema unchanged
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_gather(QueryCtxT& ctx, const Gather* gather) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!gather) {
        PGX_ERROR("Invalid Gather parameters");
        return TranslationResult{};  // Return empty result
    }

    // Access Gather-specific fields with direct field access
    const int num_workers = gather->num_workers;
    const bool single_copy = gather->single_copy;

    // Extract Gather-specific information
    if (num_workers > 0) {
    }
    if (single_copy) {
    }

    // Translate child plan - single code path for tests and production
    TranslationResult childResult;

    if (Plan* leftTree = gather->plan.lefttree) {
        childResult = translate_plan_node(ctx, leftTree);
        if (!childResult.op) {
            PGX_ERROR("Failed to translate Gather child plan");
            return childResult;  // Return empty result
        }
    } else {
        PGX_WARNING("Gather node has no child plan");
        // For unit tests, return empty result and handle gracefully
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
    auto queryFuncType = builder.getFunctionType({}, {});
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
        return input;  // Pass through unchanged
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

        // Store current builder and tuple for expression translation
        const auto tmp_ctx = QueryCtxT{
            ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.column_mappings,
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

    // Selection doesn't change the schema - pass through input's columns
    TranslationResult result;
    result.op = selectionOp;
    result.columns = input.columns;  // Inherit input's schema unchanged
    return result;
}

auto PostgreSQLASTTranslator::Impl::apply_projection_from_target_list(const QueryCtxT& ctx, const TranslationResult& input,
                                                                      const List* target_list) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!input.op || !target_list || target_list->length <= 0 || !target_list->elements) {
        return input;  // Pass through unchanged
    }

    auto inputValue = input.op->getResult(0);
    if (!inputValue) {
        PGX_ERROR("Input operation has no result");
        return input;  // Pass through unchanged
    }

    auto computedEntries = std::vector<TargetEntry*>();
    for (int i = 0; i < target_list->length; i++) {
        auto* tle = static_cast<TargetEntry*>(lfirst(&target_list->elements[i]));
        if (tle && tle->expr && tle->expr->type != T_Var) {
            computedEntries.push_back(tle);
        }
    }

    if (computedEntries.empty()) {
        return input;  // No computed columns, pass through unchanged
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

        // Translate expressions in temporary context to discover types
        {
            auto& tempRegion = tempMapOp.getPredicate();
            auto* tempBlock = &tempRegion.emplaceBlock();
            auto tupleArg = tempBlock->addArgument(mlir::relalg::TupleType::get(&context_), ctx.builder.getUnknownLoc());

            mlir::OpBuilder temp_builder(&context_);
            temp_builder.setInsertionPointToStart(tempBlock);
            auto tmp_ctx = QueryCtxT{ctx.current_stmt, temp_builder, ctx.current_module, tupleArg, ctx.column_mappings};

            for (auto* entry : computedEntries) {
                auto colName = entry->resname ? entry->resname : "col_" + std::to_string(entry->resno);
                if (colName == "?column?")
                    colName = "col_" + std::to_string(entry->resno);

                if (auto exprValue = translate_expression(tmp_ctx, entry->expr)) {
                    expressionTypes.push_back(exprValue.getType());
                    columnNames.push_back(colName);
                    Oid typeOid = exprType(reinterpret_cast<Node*>(entry->expr));
                    expressionOids.push_back(typeOid);
                }
            }
        }
        tempMapOp.erase();

        if (expressionTypes.empty()) {
            return input;  // Failed to compute types, pass through unchanged
        }
    }

    // Second pass: Create real MapOp with correct types
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
        auto tmp_ctx = QueryCtxT{ctx.current_stmt, predicate_builder, ctx.current_module, tupleArg, ctx.column_mappings};

        std::vector<mlir::Value> computedValues;
        for (auto* entry : computedEntries) {
            if (auto exprValue = translate_expression(tmp_ctx, entry->expr)) {
                computedValues.push_back(exprValue);
            }
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), computedValues);
    }

    // Build result with updated schema
    TranslationResult result;
    result.op = mapOp;
    result.columns = input.columns;
    for (size_t i = 0; i < expressionTypes.size(); i++) {
        result.columns.push_back({
            .table_name = COMPUTED_EXPRESSION_SCOPE,
            .column_name = columnNames[i],
            .type_oid = expressionOids[i],
            .typmod = -1,
            .mlir_type = expressionTypes[i],
            .nullable = true
        });
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::create_materialize_op(const QueryCtxT& context, const mlir::Value tuple_stream, const TranslationResult& translation_result) const
    -> void {
    PGX_IO(AST_TRANSLATE);
    if (!translation_result.columns.empty()) {
        auto& columnManager = context.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        std::vector<mlir::Attribute> columnRefAttrs;
        std::vector<mlir::Attribute> columnNameAttrs;

        for (const auto& column : translation_result.columns) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "MaterializeOp using result column: table=%s, name=%s, oid=%d",
                    column.table_name.c_str(), column.column_name.c_str(), column.type_oid);

            auto colRef = columnManager.createRef(column.table_name, column.column_name);
            columnRefAttrs.push_back(colRef);

            auto nameAttr = context.builder.getStringAttr(column.column_name);
            columnNameAttrs.push_back(nameAttr);
        }

        auto columnRefs = context.builder.getArrayAttr(columnRefAttrs);
        auto columnNames = context.builder.getArrayAttr(columnNameAttrs);
        auto tableType = mlir::dsa::TableType::get(&context_);

        context.builder.create<mlir::relalg::MaterializeOp>(context.builder.getUnknownLoc(), tableType, tuple_stream,
                                                            columnRefs, columnNames);
    } else {
        throw std::runtime_error("Should be impossible");
    }
}

} // namespace postgresql_ast
