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

auto PostgreSQLASTTranslator::Impl::translate_seq_scan(QueryCtxT& ctx, SeqScan* seq_scan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!seq_scan) {
        PGX_ERROR("Invalid SeqScan parameters");
        throw std::runtime_error("Invalid SeqScan parameters");
    }

    auto physical_table_name = std::string();
    auto alias_name = std::string();
    auto table_oid = InvalidOid;

    if (seq_scan->scan.scanrelid > 0) {
        physical_table_name = get_table_name_from_rte(&ctx.current_stmt, seq_scan->scan.scanrelid);
        alias_name = get_table_alias_from_rte(&ctx.current_stmt, seq_scan->scan.scanrelid);
        table_oid = get_table_oid_from_rte(&ctx.current_stmt, seq_scan->scan.scanrelid);

        if (physical_table_name.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", seq_scan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", seq_scan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    auto table_identifier = physical_table_name + TABLE_OID_SEPARATOR + std::to_string(table_oid);
    const auto TABLE_META_DATA = std::make_shared<runtime::TableMetaData>();
    TABLE_META_DATA->setNumRows(0); // Will be updated from PostgreSQL catalog
    auto table_meta_attr = mlir::relalg::TableMetaDataAttr::get(&context_, TABLE_META_DATA);

    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto unique_scope = column_manager.getUniqueScope(alias_name);
    auto column_defs = std::vector<mlir::NamedAttribute>{};
    auto column_order = std::vector<mlir::Attribute>{};
    const auto ALL_COLUMNS = get_all_table_columns_from_schema(&ctx.current_stmt, seq_scan->scan.scanrelid);

    if (!ALL_COLUMNS.empty()) {
        int varattno = 1;
        for (const auto& col_info : ALL_COLUMNS) {
            auto col_def = column_manager.createDef(unique_scope, col_info.name);

            PostgreSQLTypeMapper const type_mapper(context_);
            const mlir::Type MLIR_TYPE = type_mapper.map_postgre_sqltype(col_info.type_oid, col_info.typmod,
                                                                        col_info.nullable);
            col_def.getColumn().type = MLIR_TYPE;

            column_defs.push_back(ctx.builder.getNamedAttr(col_info.name, col_def));
            column_order.push_back(ctx.builder.getStringAttr(col_info.name));

            varattno++;
        }
    } else {
        PGX_ERROR("Could not discover table schema");
        throw std::runtime_error("Could not discover table schema");
    }

    auto columns_attr = ctx.builder.getDictionaryAttr(column_defs);
    auto column_order_attr = ctx.builder.getArrayAttr(column_order);

    const auto BASE_TABLE_OP = ctx.builder.create<mlir::relalg::BaseTableOp>(
        ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&context_),
        ctx.builder.getStringAttr(table_identifier), table_meta_attr, columns_attr, column_order_attr);

    auto result = TranslationResult();
    result.op = BASE_TABLE_OP;

    if (!seq_scan->scan.plan.targetlist || seq_scan->scan.plan.targetlist->length <= 0) {
        throw std::runtime_error("SeqScan had an empty target list");
    }

    result.columns = build_scan_columns(seq_scan->scan.plan.targetlist, ALL_COLUMNS, unique_scope);

    if (unique_scope != alias_name) {
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_seq_scan: uniqueScope != aliasName, populating varno_resolution");
        for (size_t i = 0; i < ALL_COLUMNS.size(); i++) {
            const int VARATTNO = static_cast<int>(i + 1);
            ctx.varno_resolution[std::make_pair(seq_scan->scan.scanrelid, VARATTNO)] = std::make_pair(
                unique_scope, ALL_COLUMNS[i].name);
            PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_seq_scan: varno_resolution[(%d,%d)] = ('%s','%s')",
                    seq_scan->scan.scanrelid, VARATTNO, unique_scope.c_str(), ALL_COLUMNS[i].name.c_str());
        }
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_seq_scan: uniqueScope == aliasName, NOT populating varno_resolution");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_seq_scan: final varno_resolution.size()=%zu",
            ctx.varno_resolution.size());

    // where + projection - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if (result.op && seq_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SeqScan has qual, applying selection (context has %zu InitPlans)%s",
                ctx.params.size(), ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, seq_scan->scan.plan.qual);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SeqScan: no qual (result.op=%p, plan.qual=%p)", static_cast<void*>(result.op),
                static_cast<void*>(seq_scan->scan.plan.qual));
    }

    if (result.op) {
        result = apply_projection_from_target_list(ctx, result, seq_scan->scan.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_index_scan(QueryCtxT& ctx, IndexScan* index_scan) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!index_scan) {
        PGX_ERROR("Invalid IndexScan parameters");
        throw std::runtime_error("Invalid IndexScan parameters");
    }

    auto physical_table_name = std::string();
    auto alias_name = std::string();
    auto table_oid = InvalidOid;

    if (index_scan->scan.scanrelid > 0) {
        physical_table_name = get_table_name_from_rte(&ctx.current_stmt, index_scan->scan.scanrelid);
        alias_name = get_table_alias_from_rte(&ctx.current_stmt, index_scan->scan.scanrelid);
        table_oid = get_table_oid_from_rte(&ctx.current_stmt, index_scan->scan.scanrelid);

        if (physical_table_name.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", index_scan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", index_scan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    auto table_identifier = physical_table_name + TABLE_OID_SEPARATOR + std::to_string(table_oid);
    const auto TABLE_META_DATA = std::make_shared<runtime::TableMetaData>();
    TABLE_META_DATA->setNumRows(0);
    auto table_meta_attr = mlir::relalg::TableMetaDataAttr::get(&context_, TABLE_META_DATA);

    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto unique_scope = column_manager.getUniqueScope(alias_name);
    auto column_defs = std::vector<mlir::NamedAttribute>{};
    auto column_order = std::vector<mlir::Attribute>{};
    const auto ALL_COLUMNS = get_all_table_columns_from_schema(&ctx.current_stmt, index_scan->scan.scanrelid);

    if (!ALL_COLUMNS.empty()) {
        int varattno = 1;
        for (const auto& col_info : ALL_COLUMNS) {
            auto col_def = column_manager.createDef(unique_scope, col_info.name);

            PostgreSQLTypeMapper const type_mapper(context_);
            const mlir::Type MLIR_TYPE = type_mapper.map_postgre_sqltype(col_info.type_oid, col_info.typmod,
                                                                        col_info.nullable);
            col_def.getColumn().type = MLIR_TYPE;

            column_defs.push_back(ctx.builder.getNamedAttr(col_info.name, col_def));
            column_order.push_back(ctx.builder.getStringAttr(col_info.name));

            varattno++;
        }
    } else {
        PGX_ERROR("Could not discover table schema");
        throw std::runtime_error("Could not discover table schema");
    }

    auto columns_attr = ctx.builder.getDictionaryAttr(column_defs);
    auto column_order_attr = ctx.builder.getArrayAttr(column_order);

    const auto BASE_TABLE_OP = ctx.builder.create<mlir::relalg::BaseTableOp>(
        ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&context_),
        ctx.builder.getStringAttr(table_identifier), table_meta_attr, columns_attr, column_order_attr);

    auto result = TranslationResult();
    result.op = BASE_TABLE_OP;

    if (!index_scan->scan.plan.targetlist || index_scan->scan.plan.targetlist->length <= 0) {
        throw std::runtime_error("IndexScan had an empty target list");
    }

    result.columns = build_scan_columns(index_scan->scan.plan.targetlist, ALL_COLUMNS, unique_scope);

    // populate varno_resolution for IndexScan because indexqual contains INDEX_VAR nodes
    PGX_LOG(AST_TRANSLATE, DEBUG,
            "[SCOPE_DEBUG] translate_index_scan: populating varno_resolution (uniqueScope=%s, aliasName=%s)",
            unique_scope.c_str(), alias_name.c_str());

    for (size_t i = 0; i < ALL_COLUMNS.size(); i++) {
        const int VARATTNO = static_cast<int>(i + 1);

        // Add mapping for scanrelid (regular Var and INDEX_VAR lookups)
        ctx.varno_resolution[std::make_pair(index_scan->scan.scanrelid, VARATTNO)] = std::make_pair(
            unique_scope, ALL_COLUMNS[i].name);
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_index_scan: varno_resolution[(%d,%d)] = ('%s','%s')",
                index_scan->scan.scanrelid, VARATTNO, unique_scope.c_str(), ALL_COLUMNS[i].name.c_str());
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_index_scan: final varno_resolution.size()=%zu",
            ctx.varno_resolution.size());

    if (result.op && index_scan->indexqual && index_scan->indexqual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexScan has %d indexqual predicates, applying as selection%s",
                index_scan->indexqual->length, ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, index_scan->indexqual);
    }

    if (result.op && index_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexScan has plan.qual, applying selection (context has %zu InitPlans)%s",
                ctx.params.size(), ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, index_scan->scan.plan.qual);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexScan: no plan.qual (result.op=%p, plan.qual=%p)",
                static_cast<void*>(result.op), static_cast<void*>(index_scan->scan.plan.qual));
    }

    if (result.op) {
        result = apply_projection_from_target_list(ctx, result, index_scan->scan.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_index_only_scan(QueryCtxT& ctx, IndexOnlyScan* index_only_scan)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!index_only_scan) {
        PGX_ERROR("Invalid IndexOnlyScan parameters");
        throw std::runtime_error("Invalid IndexOnlyScan parameters");
    }

    auto physical_table_name = std::string();
    auto alias_name = std::string();
    auto table_oid = InvalidOid;

    if (index_only_scan->scan.scanrelid > 0) {
        physical_table_name = get_table_name_from_rte(&ctx.current_stmt, index_only_scan->scan.scanrelid);
        alias_name = get_table_alias_from_rte(&ctx.current_stmt, index_only_scan->scan.scanrelid);
        table_oid = get_table_oid_from_rte(&ctx.current_stmt, index_only_scan->scan.scanrelid);

        if (physical_table_name.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", index_only_scan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", index_only_scan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    auto table_identifier = physical_table_name + TABLE_OID_SEPARATOR + std::to_string(table_oid);
    const auto TABLE_META_DATA = std::make_shared<runtime::TableMetaData>();
    TABLE_META_DATA->setNumRows(0);
    auto table_meta_attr = mlir::relalg::TableMetaDataAttr::get(&context_, TABLE_META_DATA);

    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto unique_scope = column_manager.getUniqueScope(alias_name);
    auto column_defs = std::vector<mlir::NamedAttribute>{};
    auto column_order = std::vector<mlir::Attribute>{};
    const auto ALL_COLUMNS = get_all_table_columns_from_schema(&ctx.current_stmt, index_only_scan->scan.scanrelid);

    if (!ALL_COLUMNS.empty()) {
        int varattno = 1;
        for (const auto& col_info : ALL_COLUMNS) {
            auto col_def = column_manager.createDef(unique_scope, col_info.name);

            PostgreSQLTypeMapper const type_mapper(context_);
            const mlir::Type MLIR_TYPE = type_mapper.map_postgre_sqltype(col_info.type_oid, col_info.typmod,
                                                                        col_info.nullable);
            col_def.getColumn().type = MLIR_TYPE;

            column_defs.push_back(ctx.builder.getNamedAttr(col_info.name, col_def));
            column_order.push_back(ctx.builder.getStringAttr(col_info.name));

            varattno++;
        }
    } else {
        PGX_ERROR("Could not discover table schema");
        throw std::runtime_error("Could not discover table schema");
    }

    auto columns_attr = ctx.builder.getDictionaryAttr(column_defs);
    auto column_order_attr = ctx.builder.getArrayAttr(column_order);

    const auto BASE_TABLE_OP = ctx.builder.create<mlir::relalg::BaseTableOp>(
        ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&context_),
        ctx.builder.getStringAttr(table_identifier), table_meta_attr, columns_attr, column_order_attr);

    auto result = TranslationResult();
    result.op = BASE_TABLE_OP;

    if (!index_only_scan->scan.plan.targetlist || index_only_scan->scan.plan.targetlist->length <= 0) {
        throw std::runtime_error("IndexOnlyScan had an empty target list");
    }

    result.columns = build_scan_columns(index_only_scan->scan.plan.targetlist, ALL_COLUMNS, unique_scope);

    // populate varno_resolution for IndexOnlyScan because indexqual/recheckqual may contain INDEX_VAR nodes
    PGX_LOG(AST_TRANSLATE, DEBUG,
            "[SCOPE_DEBUG] translate_index_only_scan: populating varno_resolution (uniqueScope=%s, aliasName=%s)",
            unique_scope.c_str(), alias_name.c_str());

    for (size_t i = 0; i < ALL_COLUMNS.size(); i++) {
        const int VARATTNO = static_cast<int>(i + 1);

        // Add mapping for scanrelid (regular Var and INDEX_VAR lookups)
        ctx.varno_resolution[std::make_pair(index_only_scan->scan.scanrelid, VARATTNO)] = std::make_pair(
            unique_scope, ALL_COLUMNS[i].name);
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_index_only_scan: varno_resolution[(%d,%d)] = ('%s','%s')",
                index_only_scan->scan.scanrelid, VARATTNO, unique_scope.c_str(), ALL_COLUMNS[i].name.c_str());
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_index_only_scan: final varno_resolution.size()=%zu",
            ctx.varno_resolution.size());

    // Apply indexqual (index access conditions)
    if (result.op && index_only_scan->indexqual && index_only_scan->indexqual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexOnlyScan has %d indexqual predicates, applying as selection%s",
                index_only_scan->indexqual->length, ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, index_only_scan->indexqual);
    }

    // Apply recheckqual (lossy index recheck conditions)
    if (result.op && index_only_scan->recheckqual && index_only_scan->recheckqual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexOnlyScan has %d recheckqual predicates, applying as selection%s",
                index_only_scan->recheckqual->length, ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, index_only_scan->recheckqual);
    }

    // Apply plan.qual (additional heap-level conditions)
    if (result.op && index_only_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexOnlyScan has plan.qual, applying selection (context has %zu InitPlans)%s",
                ctx.params.size(), ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, index_only_scan->scan.plan.qual);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "IndexOnlyScan: no plan.qual (result.op=%p, plan.qual=%p)",
                static_cast<void*>(result.op), static_cast<void*>(index_only_scan->scan.plan.qual));
    }

    if (result.op) {
        result = apply_projection_from_target_list(ctx, result, index_only_scan->scan.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_bitmap_heap_scan(QueryCtxT& ctx, BitmapHeapScan* bitmap_scan)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!bitmap_scan) {
        PGX_ERROR("Invalid BitmapHeapScan parameters");
        throw std::runtime_error("Invalid BitmapHeapScan parameters");
    }

    auto physical_table_name = std::string();
    auto alias_name = std::string();
    auto table_oid = InvalidOid;

    if (bitmap_scan->scan.scanrelid > 0) {
        physical_table_name = get_table_name_from_rte(&ctx.current_stmt, bitmap_scan->scan.scanrelid);
        alias_name = get_table_alias_from_rte(&ctx.current_stmt, bitmap_scan->scan.scanrelid);
        table_oid = get_table_oid_from_rte(&ctx.current_stmt, bitmap_scan->scan.scanrelid);

        if (physical_table_name.empty()) {
            PGX_ERROR("Could not resolve table name for scanrelid: %d", bitmap_scan->scan.scanrelid);
            throw std::runtime_error("Could not resolve table name for scanrelid");
        }
    } else {
        PGX_ERROR("Invalid scan relation ID: %d", bitmap_scan->scan.scanrelid);
        throw std::runtime_error("Could not resolve table name for scanrelid");
    }

    auto table_identifier = physical_table_name + TABLE_OID_SEPARATOR + std::to_string(table_oid);
    const auto TABLE_META_DATA = std::make_shared<runtime::TableMetaData>();
    TABLE_META_DATA->setNumRows(0);
    auto table_meta_attr = mlir::relalg::TableMetaDataAttr::get(&context_, TABLE_META_DATA);

    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    auto unique_scope = column_manager.getUniqueScope(alias_name);
    auto column_defs = std::vector<mlir::NamedAttribute>{};
    auto column_order = std::vector<mlir::Attribute>{};
    const auto ALL_COLUMNS = get_all_table_columns_from_schema(&ctx.current_stmt, bitmap_scan->scan.scanrelid);

    if (!ALL_COLUMNS.empty()) {
        int varattno = 1;
        for (const auto& col_info : ALL_COLUMNS) {
            auto col_def = column_manager.createDef(unique_scope, col_info.name);

            PostgreSQLTypeMapper const type_mapper(context_);
            const mlir::Type MLIR_TYPE = type_mapper.map_postgre_sqltype(col_info.type_oid, col_info.typmod,
                                                                        col_info.nullable);
            col_def.getColumn().type = MLIR_TYPE;

            column_defs.push_back(ctx.builder.getNamedAttr(col_info.name, col_def));
            column_order.push_back(ctx.builder.getStringAttr(col_info.name));

            varattno++;
        }
    } else {
        PGX_ERROR("Could not discover table schema");
        throw std::runtime_error("Could not discover table schema");
    }

    auto columns_attr = ctx.builder.getDictionaryAttr(column_defs);
    auto column_order_attr = ctx.builder.getArrayAttr(column_order);

    const auto BASE_TABLE_OP = ctx.builder.create<mlir::relalg::BaseTableOp>(
        ctx.builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&context_),
        ctx.builder.getStringAttr(table_identifier), table_meta_attr, columns_attr, column_order_attr);

    auto result = TranslationResult();
    result.op = BASE_TABLE_OP;

    if (!bitmap_scan->scan.plan.targetlist || bitmap_scan->scan.plan.targetlist->length <= 0) {
        throw std::runtime_error("BitmapHeapScan had an empty target list");
    }

    result.columns = build_scan_columns(bitmap_scan->scan.plan.targetlist, ALL_COLUMNS, unique_scope);

    PGX_LOG(AST_TRANSLATE, DEBUG,
            "[SCOPE_DEBUG] translate_bitmap_heap_scan: populating varno_resolution (uniqueScope=%s, aliasName=%s)",
            unique_scope.c_str(), alias_name.c_str());

    for (size_t i = 0; i < ALL_COLUMNS.size(); i++) {
        const int VARATTNO = static_cast<int>(i + 1);

        ctx.varno_resolution[std::make_pair(bitmap_scan->scan.scanrelid, VARATTNO)] = std::make_pair(
            unique_scope, ALL_COLUMNS[i].name);
        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[SCOPE_DEBUG] translate_bitmap_heap_scan: varno_resolution[(%d,%d)] = ('%s','%s')",
                bitmap_scan->scan.scanrelid, VARATTNO, unique_scope.c_str(), ALL_COLUMNS[i].name.c_str());
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_bitmap_heap_scan: final varno_resolution.size()=%zu",
            ctx.varno_resolution.size());

    if (result.op && bitmap_scan->bitmapqualorig && bitmap_scan->bitmapqualorig->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "BitmapHeapScan has %d bitmapqualorig predicates, applying as selection%s",
                bitmap_scan->bitmapqualorig->length, ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, bitmap_scan->bitmapqualorig);
    }

    if (result.op && bitmap_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "BitmapHeapScan has plan.qual, applying selection (context has %zu InitPlans)%s",
                ctx.params.size(), ctx.outer_result.has_value() ? " (parameterized)" : "");
        result = apply_selection_from_qual_with_columns(ctx, result, bitmap_scan->scan.plan.qual);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "BitmapHeapScan: no plan.qual (result.op=%p, plan.qual=%p)",
                static_cast<void*>(result.op), static_cast<void*>(bitmap_scan->scan.plan.qual));
    }

    if (result.op) {
        result = apply_projection_from_target_list(ctx, result, bitmap_scan->scan.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_subquery_scan(QueryCtxT& ctx, SubqueryScan* subquery_scan)
    -> TranslationResult {
    PGX_IO(AST_TRANSLATE);

    if (!subquery_scan || !subquery_scan->subplan) {
        PGX_ERROR("Invalid SubqueryScan parameters");
        throw std::runtime_error("Invalid SubqueryScan parameters");
    }

    const auto SCANRELID = subquery_scan->scan.scanrelid;
    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating SubqueryScan with scanrelid=%d", SCANRELID);

    auto result = translate_plan_node(ctx, subquery_scan->subplan);

    if (!result.op) {
        PGX_ERROR("Failed to translate SubqueryScan subplan");
        throw std::runtime_error("Failed to translate SubqueryScan subplan");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan subplan translated with %zu columns", result.columns.size());

    if (SCANRELID > 0 && subquery_scan->scan.plan.targetlist) {
        List* const targetlist = subquery_scan->scan.plan.targetlist;

        const std::string SUBQUERY_ALIAS = get_table_alias_from_rte(&ctx.current_stmt, SCANRELID);

        std::vector<TranslationResult::ColumnSchema> subplan_columns = result.columns;
        result.columns.clear();

        ListCell* lc = nullptr;
        int output_attno = 1;

        foreach (lc, targetlist) {
            auto* tle = static_cast<TargetEntry*>(lfirst(lc));

            if (!tle->expr) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan: Skipping targetlist entry with no expression at attno=%d",
                        output_attno);
                output_attno++;
                continue;
            }

            if (IsA(tle->expr, Var)) {
                auto* var = reinterpret_cast<Var*>(tle->expr);

                if (var->varattno > 0 && var->varattno <= static_cast<int>(subplan_columns.size())) {
                    const auto& col = subplan_columns[var->varattno - 1];

                    ctx.varno_resolution[std::make_pair(SCANRELID, output_attno)] = std::make_pair(col.table_name,
                                                                                                      col.column_name);
                    result.columns.push_back(col);
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Mapped SubqueryScan: varno=%d, attno=%d -> subplan column %d (@%s::@%s)", SCANRELID,
                            output_attno, var->varattno, col.table_name.c_str(), col.column_name.c_str());
                }
            } else {
                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan: Processing complex expression at attno=%d", output_attno);
                const std::string COL_NAME = tle->resname ? tle->resname : "expr_" + std::to_string(output_attno);

                TranslationResult expr_context;
                expr_context.op = result.op;
                expr_context.columns = subplan_columns;
                for (size_t i = 0; i < subplan_columns.size(); ++i) {
                    const auto& col = subplan_columns[i];
                    ctx.varno_resolution[std::make_pair(SCANRELID, i + 1)] = std::make_pair(col.table_name,
                                                                                                    col.column_name);
                }
                auto stream_result = translate_expression_for_stream(ctx, tle->expr, expr_context, COL_NAME);
                verify_and_print(stream_result.stream);
                result.op = stream_result.stream.getDefiningOp();
                // ReSharper disable once CppDFAUnusedValue,CppDFAUnreadVariable
                Oid const type_oid = exprType(reinterpret_cast<Node*>(tle->expr));
                // ReSharper disable once CppDFAUnusedValue,CppDFAUnreadVariable
                int32_t const typmod = exprTypmod(reinterpret_cast<Node*>(tle->expr));
                mlir::Type const expr_type = stream_result.stream.getType();
                // ReSharper disable once CppDFAUnreadVariable,CppDFAUnusedValue
                bool const nullable = mlir::isa<mlir::db::NullableType>(expr_type);

                result.columns.push_back(
                    {stream_result.table_name, stream_result.column_name, type_oid, typmod, expr_type, nullable});

                ctx.varno_resolution[std::make_pair(SCANRELID, output_attno)] = std::make_pair(
                    stream_result.table_name, stream_result.column_name);

                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan expression: varno=%d, attno=%d -> @%s::@%s", SCANRELID,
                        output_attno, stream_result.table_name.c_str(), stream_result.column_name.c_str());
            }
            output_attno++;
        }

        if (!SUBQUERY_ALIAS.empty()) {
            result.current_scope = SUBQUERY_ALIAS;
        }
    }

    if (result.op && subquery_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan has qual, applying selection (context has %zu InitPlans)",
                ctx.params.size());
        result = apply_selection_from_qual_with_columns(ctx, result, subquery_scan->scan.plan.qual);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_cte_scan(QueryCtxT& ctx, const CteScan* cte_scan) -> TranslationResult {
    // CteScan is a bit confusing. It has a plan inside of it, but these plans are evaluated at InitPlan time,
    // so we just need to read out of the target list here.
    PGX_IO(AST_TRANSLATE);

    if (!cte_scan) {
        PGX_ERROR("Invalid CteScan parameters");
        throw std::runtime_error("Invalid CteScan parameters");
    }

    const auto CTE_PARAM = cte_scan->cteParam;
    const auto CTE_PLAN_ID = cte_scan->ctePlanId;
    const auto SCANRELID = cte_scan->scan.scanrelid;

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating CteScan with cteParam=%d, ctePlanId=%d, scanrelid=%d", CTE_PARAM,
            CTE_PLAN_ID, SCANRELID);

    const auto IT = ctx.initplan_results.find(CTE_PARAM);
    if (IT == ctx.initplan_results.end()) {
        PGX_ERROR("CTE InitPlan result not found for cteParam=%d", CTE_PARAM);
        throw std::runtime_error("CTE InitPlan result not found");
    }
    TranslationResult result = IT->second;

    if (!result.op) {
        PGX_ERROR("CTE InitPlan has no operation for cteParam=%d", CTE_PARAM);
        throw std::runtime_error("CTE InitPlan has no operation");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Found CTE InitPlan result with %zu columns", result.columns.size());

    if (SCANRELID <= 0 || !cte_scan->scan.plan.targetlist) {
        return result;
    }
    List* const targetlist = cte_scan->scan.plan.targetlist;

    const std::string CTE_ALIAS = get_table_alias_from_rte(&ctx.current_stmt, SCANRELID);
    const bool NEEDS_PROJECTION = !CTE_ALIAS.empty();

    std::vector<mlir::Attribute> projection_columns;
    std::vector<TranslationResult::ColumnSchema> new_columns;
    auto& column_manager = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

    ListCell* lc = nullptr;
    int output_attno = 1;

    foreach (lc, targetlist) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));

        if (tle->expr && IsA(tle->expr, Var)) {
            auto* var = reinterpret_cast<Var*>(tle->expr);

            if (var->varattno > 0 && var->varattno <= static_cast<int>(result.columns.size())) {
                // ReSharper disable once CppUseStructuredBinding
                const auto& col = result.columns[var->varattno - 1];

                if (NEEDS_PROJECTION && tle->resname) {
                    const std::string NEW_COL_NAME = tle->resname;
                    auto col_ref = column_manager.createDef(CTE_ALIAS, NEW_COL_NAME);
                    col_ref.getColumn().type = col.mlir_type;
                    projection_columns.push_back(col_ref);

                    new_columns.push_back({CTE_ALIAS, NEW_COL_NAME, col.type_oid, col.typmod, col.mlir_type, col.nullable});

                    ctx.varno_resolution[std::make_pair(SCANRELID, var->varattno)] = std::make_pair(CTE_ALIAS,
                                                                                                      NEW_COL_NAME);

                    PGX_LOG(AST_TRANSLATE, DEBUG, "CteScan column aliasing: varno=%d, attno=%d: @%s::@%s -> @%s::@%s",
                            SCANRELID, var->varattno, col.table_name.c_str(), col.column_name.c_str(), CTE_ALIAS.c_str(),
                            NEW_COL_NAME.c_str());
                } else {
                    new_columns.push_back(col);
                    ctx.varno_resolution[std::make_pair(SCANRELID, var->varattno)] = std::make_pair(col.table_name,
                                                                                                      col.column_name);
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Mapped CteScan: varno=%d, attno=%d -> CTE column %d (@%s::@%s)",
                            SCANRELID, var->varattno, var->varattno, col.table_name.c_str(), col.column_name.c_str());
                }
            }
        }
        output_attno++;
    }

    if (!new_columns.empty()) {
        result.columns = new_columns;
    }

    if (NEEDS_PROJECTION && !projection_columns.empty()) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Creating projection with %zu aliased columns for CTE '%s'",
                projection_columns.size(), CTE_ALIAS.c_str());

        auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.builder.getContext());
        auto projection_op = ctx.builder.create<mlir::relalg::ProjectionOp>(
            ctx.builder.getUnknownLoc(), tuple_stream_type,
            mlir::relalg::SetSemanticAttr::get(ctx.builder.getContext(), mlir::relalg::SetSemantic::all),
            result.op->getResult(0), ctx.builder.getArrayAttr(projection_columns));

        result.op = projection_op.getOperation();
        result.current_scope = CTE_ALIAS;
    }

    if (result.op && cte_scan->scan.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "CteScan has qual, applying selection (context has %zu InitPlans)",
                ctx.params.size());
        result = apply_selection_from_qual_with_columns(ctx, result, cte_scan->scan.plan.qual);
    }

    return result;
}

} // namespace postgresql_ast
