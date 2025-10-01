#include "translator_internals.h"

extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "nodes/nodeFuncs.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
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

auto PostgreSQLASTTranslator::Impl::translate_seq_scan(QueryCtxT& ctx, SeqScan* seqScan) -> TranslationResult {
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

    auto result = TranslationResult();
    result.op = baseTableOp;

    if (!seqScan->scan.plan.targetlist || seqScan->scan.plan.targetlist->length <= 0) {
        throw std::runtime_error("SeqScan had an empty target list");
    }

    ListCell* lc;
    foreach (lc, seqScan->scan.plan.targetlist) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle || tle->resjunk)
            continue;
        if (tle->expr && IsA(tle->expr, Var)) {
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

    if (uniqueScope != aliasName) {
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_seq_scan: uniqueScope != aliasName, populating varno_resolution");
        for (size_t i = 0; i < allColumns.size(); i++) {
            const int varattno = static_cast<int>(i + 1);
            result.varno_resolution[std::make_pair(seqScan->scan.scanrelid, varattno)] = std::make_pair(
                uniqueScope, allColumns[i].name);
            PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_seq_scan: varno_resolution[(%d,%d)] = ('%s','%s')",
                    seqScan->scan.scanrelid, varattno, uniqueScope.c_str(), allColumns[i].name.c_str());
        }
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_seq_scan: uniqueScope == aliasName, NOT populating varno_resolution");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_seq_scan: final varno_resolution.size()=%zu",
            result.varno_resolution.size());

    // Apply qual (WHERE clause) if present
    if (result.op && seqScan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SeqScan has qual, applying selection (context has %zu InitPlans)",
                ctx.init_plan_results.size());
        result = apply_selection_from_qual_with_columns(ctx, result, seqScan->scan.plan.qual, nullptr, nullptr);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SeqScan: no qual (result.op=%p, plan.qual=%p)",
                static_cast<void*>(result.op), static_cast<void*>(seqScan->scan.plan.qual));
    }

    // Apply projection (computed expressions) if present
    if (result.op && seqScan->scan.plan.targetlist) {
        result = apply_projection_from_target_list(ctx, result, seqScan->scan.plan.targetlist);
    }

    return result;
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

        ListCell* lc;
        int output_attno = 1;

        foreach (lc, targetlist) {
            auto* tle = reinterpret_cast<TargetEntry*>(lfirst(lc));
            if (tle->resjunk) {
                continue;
            }

            if (!tle->expr) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan: Skipping targetlist entry with no expression at attno=%d",
                        output_attno);
                output_attno++;
                continue;
            }

            if (IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(result.columns.size())) {
                    const auto& col = result.columns[var->varattno - 1];

                    result.varno_resolution[std::make_pair(scanrelid, output_attno)] = std::make_pair(
                        col.table_name, col.column_name);

                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Mapped SubqueryScan: varno=%d, attno=%d -> subplan column %d (@%s::@%s)", scanrelid,
                            output_attno, var->varattno, col.table_name.c_str(), col.column_name.c_str());
                }
            } else {
                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan: Processing complex expression at attno=%d", output_attno);
                const std::string col_name = tle->resname ? tle->resname : "expr_" + std::to_string(output_attno);

                TranslationResult exprContext = result;
                for (size_t i = 0; i < result.columns.size(); ++i) {
                    const auto& col = result.columns[i];
                    exprContext.varno_resolution[std::make_pair(scanrelid, i + 1)] = std::make_pair(col.table_name,
                                                                                                    col.column_name);
                }
                auto streamResult = translate_expression_for_stream(ctx, tle->expr, exprContext, col_name);
                verify_and_print(streamResult.stream);
                result.op = streamResult.stream.getDefiningOp();
                Oid type_oid = exprType(reinterpret_cast<Node*>(tle->expr));
                int32_t typmod = exprTypmod(reinterpret_cast<Node*>(tle->expr));
                mlir::Type exprType = streamResult.stream.getType();
                bool nullable = mlir::isa<mlir::db::NullableType>(exprType);

                result.columns.push_back(
                    {streamResult.table_name, streamResult.column_name, type_oid, typmod, exprType, nullable});

                result.varno_resolution[std::make_pair(scanrelid, output_attno)] = std::make_pair(
                    streamResult.table_name, streamResult.column_name);

                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan expression: varno=%d, attno=%d -> @%s::@%s", scanrelid,
                        output_attno, streamResult.table_name.c_str(), streamResult.column_name.c_str());
            }
            output_attno++;
        }

        if (!subquery_alias.empty()) {
            result.current_scope = subquery_alias;
        }
    }

    if (result.op && subqueryScan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan has qual, applying selection (context has %zu InitPlans)",
                ctx.init_plan_results.size());
        result = apply_selection_from_qual_with_columns(ctx, result, subqueryScan->scan.plan.qual, nullptr, nullptr);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_cte_scan(QueryCtxT& ctx, CteScan* cteScan) -> TranslationResult {
    // CteScan is a bit confusing. It has a plan inside of it, but these plans are evaluated at InitPlan time,
    // so we just need to read out of the target list here.
    PGX_IO(AST_TRANSLATE);

    if (!cteScan) {
        PGX_ERROR("Invalid CteScan parameters");
        throw std::runtime_error("Invalid CteScan parameters");
    }

    const auto cteParam = cteScan->cteParam;
    const auto ctePlanId = cteScan->ctePlanId;
    const auto scanrelid = cteScan->scan.scanrelid;

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating CteScan with cteParam=%d, ctePlanId=%d, scanrelid=%d", cteParam,
            ctePlanId, scanrelid);

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

    if (scanrelid <= 0 || !cteScan->scan.plan.targetlist) {
        return result;
    }
    List* targetlist = cteScan->scan.plan.targetlist;

    const std::string cte_alias = get_table_alias_from_rte(&ctx.current_stmt, scanrelid);
    const bool needs_projection = !cte_alias.empty();

    std::vector<mlir::Attribute> projectionColumns;
    std::vector<TranslationResult::ColumnSchema> newColumns;
    auto& columnManager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    ListCell* lc;
    int output_attno = 1;

    foreach (lc, targetlist) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle->resjunk)
            continue;

        if (tle->expr && IsA(tle->expr, Var)) {
            auto* var = reinterpret_cast<Var*>(tle->expr);

            if (var->varattno > 0 && var->varattno <= static_cast<int>(result.columns.size())) {
                const auto& col = result.columns[var->varattno - 1];

                if (needs_projection && tle->resname) {
                    const std::string new_col_name = tle->resname;
                    auto colRef = columnManager.createDef(cte_alias, new_col_name);
                    colRef.getColumn().type = col.mlir_type;
                    projectionColumns.push_back(colRef);

                    newColumns.push_back({cte_alias, new_col_name, col.type_oid, col.typmod, col.mlir_type, col.nullable});

                    result.varno_resolution[std::make_pair(scanrelid, output_attno)] = std::make_pair(cte_alias,
                                                                                                      new_col_name);

                    PGX_LOG(AST_TRANSLATE, DEBUG, "CteScan column aliasing: varno=%d, attno=%d: @%s::@%s -> @%s::@%s",
                            scanrelid, output_attno, col.table_name.c_str(), col.column_name.c_str(), cte_alias.c_str(),
                            new_col_name.c_str());
                } else {
                    result.varno_resolution[std::make_pair(scanrelid, output_attno)] = std::make_pair(col.table_name,
                                                                                                      col.column_name);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Mapped CteScan: varno=%d, attno=%d -> CTE column %d (@%s::@%s)",
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
            ctx.builder.getUnknownLoc(), tupleStreamType,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
            result.op->getResult(0), ctx.builder.getArrayAttr(projectionColumns));

        result.op = projectionOp.getOperation();
        result.columns = newColumns;
        result.current_scope = cte_alias;
    }

    if (result.op && cteScan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "CteScan has qual, applying selection (context has %zu InitPlans)",
                ctx.init_plan_results.size());
        result = apply_selection_from_qual_with_columns(ctx, result, cteScan->scan.plan.qual, nullptr, nullptr);
    }

    return result;
}

} // namespace postgresql_ast
