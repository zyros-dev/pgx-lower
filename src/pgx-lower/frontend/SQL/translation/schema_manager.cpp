#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"
#include "catalog/pg_operator.h"
#include "catalog/namespace.h"
#include "access/table.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "utils/syscache.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

namespace postgresql_ast {

using namespace pgx_lower::frontend::sql::constants;

auto get_table_name_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!current_planned_stmt || !current_planned_stmt->rtable || varno <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, varno);
        throw std::runtime_error("Invalid RTE");
    }

    if (varno > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", varno, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid RTE");
    }

    const auto rte = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte || rte->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for varno %d", varno);
        throw std::runtime_error("Invalid RTE");
    }

#ifdef BUILDING_UNIT_TESTS
    return std::string(UNIT_TEST_TABLE_PREFIX) + std::to_string(varno);
#else
    char* relname = get_rel_name(rte->relid);
    std::string tableName = relname ? relname : ("unknown_table_" + std::to_string(varno));

    return tableName;
#endif
}

auto get_table_alias_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!current_planned_stmt || !current_planned_stmt->rtable || varno <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, varno);
        throw std::runtime_error("Invalid RTE");
    }

    if (varno > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", varno, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid RTE");
    }

    const auto rte = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte) {
        PGX_ERROR("Invalid RTE for varno %d", varno);
        throw std::runtime_error("Invalid RTE");
    }

#ifdef BUILDING_UNIT_TESTS
    return std::string(UNIT_TEST_TABLE_PREFIX) + std::to_string(varno);
#else
    if (rte->eref && rte->eref->aliasname) {
        return std::string(rte->eref->aliasname);
    }

    if (rte->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for varno %d", varno);
        throw std::runtime_error("Invalid RTE");
    }

    char* relname = get_rel_name(rte->relid);
    return relname ? relname : ("unknown_table_" + std::to_string(varno));
#endif
}

auto get_column_name_from_schema(const PlannedStmt* currentPlannedStmt, const int varno, const AttrNumber varattno)
    -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO || varattno <= INVALID_VARATTNO) {
        PGX_ERROR("Cannot access schema for column: varno=%d varattno=%d", varno, varattno);
        throw std::runtime_error("Invalid - read logs");
    }

    if (varno > list_length(currentPlannedStmt->rtable)) {
        PGX_ERROR("varno exceeds rtable length");
        throw std::runtime_error("Invalid - read logs");
    }

    const auto rte = static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte) {
        PGX_ERROR("Invalid RTE for column lookup: varno=%d", varno);
        throw std::runtime_error("Invalid - read logs");
    }

#ifdef BUILDING_UNIT_TESTS
    if (varattno == 1)
        return "id";
    if (varattno == 2)
        return "val1";
    if (varattno == 3)
        return "val2";
    return "col_" + std::to_string(varattno);
#else
    if (rte->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for column lookup: varno=%d has no relid (CTE/subquery should use varno_resolution)", varno);
        throw std::runtime_error("Invalid - read logs");
    }

    char* attname = get_attname(rte->relid, varattno, PG_ATTNAME_NOT_MISSING_OK);
    std::string columnName = attname ? attname : ("col_" + std::to_string(varattno));

    return columnName;
#endif
}

auto get_table_oid_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> Oid {
    PGX_IO(AST_TRANSLATE);
    using namespace pgx_lower::frontend::sql::constants;
    if (!current_planned_stmt || !current_planned_stmt->rtable || varno <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, varno);
        throw std::runtime_error("Invalid - read logs");
    }

    if (varno > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", varno, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid - read logs");
    }

    const auto rte = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte) {
        PGX_ERROR("Invalid RTE for varno %d", varno);
        throw std::runtime_error("Invalid - read logs");
    }

    return rte->relid;
}

auto is_column_nullable(const PlannedStmt* currentPlannedStmt, const int varno, const AttrNumber varattno) -> bool {
    PGX_IO(AST_TRANSLATE);

    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO || varattno <= INVALID_VARATTNO) {
        return true;
    }

#ifdef BUILDING_UNIT_TESTS
    return true;
#else
    if (varno > list_length(currentPlannedStmt->rtable)) {
        return true;
    }

    const auto rte = static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));
    if (!rte || rte->relid == InvalidOid) {
        return true;
    }

    const auto rel = table_open(rte->relid, AccessShareLock);
    if (!rel) {
        return true;
    }

    const auto tupleDesc = RelationGetDescr(rel);
    if (!tupleDesc) {
        table_close(rel, AccessShareLock);
        return true;
    }

    const int attrIndex = varattno - 1;
    if (attrIndex < 0 || attrIndex >= tupleDesc->natts) {
        table_close(rel, AccessShareLock);
        return true;
    }

    const Form_pg_attribute attr = TupleDescAttr(tupleDesc, attrIndex);
    const bool nullable = !attr->attnotnull;

    table_close(rel, AccessShareLock);
    return nullable;
#endif
}

auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, const int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo> {
    PGX_IO(AST_TRANSLATE);
    std::vector<pgx_lower::frontend::sql::ColumnInfo> columns;

#ifdef BUILDING_UNIT_TESTS
    columns.emplace_back("id", INT4OID, INVALID_TYPMOD, UNIT_TEST_COLUMN_NOT_NULL);
    return columns;
#else
    if (!current_planned_stmt || !current_planned_stmt->rtable || scanrelid <= 0) {
        PGX_ERROR("Cannot access rtable for scanrelid %d", scanrelid);
        throw std::runtime_error("Invalid - read logs");
    }

    if (scanrelid > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("scanrelid exceeds rtable length");
        throw std::runtime_error("Invalid - read logs");
    }

    const auto rte = static_cast<RangeTblEntry*>(
        list_nth(current_planned_stmt->rtable, scanrelid - POSTGRESQL_VARNO_OFFSET));

    if (!rte || rte->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for table schema discovery");
        throw std::runtime_error("Invalid - read logs");
    }

    const Relation rel = table_open(rte->relid, AccessShareLock);
    if (!rel) {
        PGX_ERROR("Failed to open relation %d", rte->relid);
        throw std::runtime_error("Invalid - read logs");
    }

    const TupleDesc tupleDesc = RelationGetDescr(rel);
    if (!tupleDesc) {
        PGX_ERROR("Failed to get tuple descriptor");
        table_close(rel, AccessShareLock);
        throw std::runtime_error("Invalid - read logs");
    }

    for (int i = 0; i < tupleDesc->natts; i++) {
        const Form_pg_attribute attr = TupleDescAttr(tupleDesc, i);
        if (attr->attisdropped) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Skipping attr");
            continue;
        }

        std::string colName = NameStr(attr->attname);
        Oid colType = attr->atttypid;
        int32_t typmod = attr->atttypmod;
        bool nullable = !attr->attnotnull;

        columns.push_back({colName, colType, typmod, nullable});
    }

    table_close(rel, AccessShareLock);

    PGX_LOG(AST_TRANSLATE, DEBUG, "Discovered %zu columns for scanrelid %d", columns.size(), scanrelid);
    return columns;
#endif
}

} // namespace postgresql_ast