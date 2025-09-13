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

auto get_table_name_from_rte(PlannedStmt* currentPlannedStmt, int varno) -> std::string {
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO) {
        PGX_WARNING("Cannot access rtable: currentPlannedStmt=%p varno=%d", currentPlannedStmt, varno);
        return "unknown_table_" + std::to_string(varno);
    }

    // Get RangeTblEntry from rtable using varno (1-based index)
    if (varno > list_length(currentPlannedStmt->rtable)) {
        PGX_WARNING("varno %d exceeds rtable length %d", varno, list_length(currentPlannedStmt->rtable));
        return "unknown_table_" + std::to_string(varno);
    }

    RangeTblEntry* rte =
        static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte || rte->relid == InvalidOid) {
        PGX_WARNING("Invalid RTE for varno %d", varno);
        return "unknown_table_" + std::to_string(varno);
    }

#ifdef BUILDING_UNIT_TESTS
    // In unit tests, use dynamic fallback based on varno
    return std::string(UNIT_TEST_TABLE_PREFIX) + std::to_string(varno);
#else
    char* relname = get_rel_name(rte->relid);
    std::string tableName = relname ? relname : ("unknown_table_" + std::to_string(varno));

    return tableName;
#endif
}

auto get_table_oid_from_rte(PlannedStmt* currentPlannedStmt, int varno) -> Oid {
    using namespace pgx_lower::frontend::sql::constants;
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO) {
        PGX_WARNING("Cannot access rtable: currentPlannedStmt=%p varno=%d", currentPlannedStmt, varno);
        return InvalidOid;
    }

    // Get RangeTblEntry from rtable using varno (1-based index)
    if (varno > list_length(currentPlannedStmt->rtable)) {
        PGX_WARNING("varno %d exceeds rtable length %d", varno, list_length(currentPlannedStmt->rtable));
        return InvalidOid;
    }

    RangeTblEntry* rte =
        static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte) {
        PGX_WARNING("Invalid RTE for varno %d", varno);
        return InvalidOid;
    }

    // Return the actual PostgreSQL relation OID
    return rte->relid;
}

auto get_column_name_from_schema(const PlannedStmt* currentPlannedStmt, int varno, AttrNumber varattno) -> std::string {
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO || varattno <= INVALID_VARATTNO) {
        PGX_WARNING("Cannot access schema for column: varno=%d varattno=%d", varno, varattno);
        return "col_" + std::to_string(varattno);
    }

    // Get RangeTblEntry
    if (varno > list_length(currentPlannedStmt->rtable)) {
        PGX_WARNING("varno exceeds rtable length");
        return "col_" + std::to_string(varattno);
    }

    RangeTblEntry* rte =
        static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));

    if (!rte || rte->relid == InvalidOid) {
        PGX_WARNING("Invalid RTE for column lookup");
        return "col_" + std::to_string(varattno);
    }

#ifdef BUILDING_UNIT_TESTS
    // In unit test environment, use hardcoded column names for test_arithmetic table
    if (varattno == FIRST_COLUMN_ATTNO)
        return "id";
    if (varattno == SECOND_COLUMN_ATTNO)
        return "val1";
    if (varattno == THIRD_COLUMN_ATTNO)
        return "val2";
    return "col_" + std::to_string(varattno);
#else
    // Get column name from PostgreSQL catalog (only in PostgreSQL environment)
    char* attname = get_attname(rte->relid, varattno, PG_ATTNAME_NOT_MISSING_OK);
    std::string columnName = attname ? attname : ("col_" + std::to_string(varattno));

    return columnName;
#endif
}

auto get_all_table_columns_from_schema(const PlannedStmt* currentPlannedStmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo> {
    std::vector<pgx_lower::frontend::sql::ColumnInfo> columns;

#ifdef BUILDING_UNIT_TESTS
    // In unit test environment, return hardcoded schema for test_arithmetic table
    columns.emplace_back("id", INT4OID, INVALID_TYPMOD, UNIT_TEST_COLUMN_NOT_NULL);
    return columns;
#else
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || scanrelid <= 0) {
        PGX_WARNING("Cannot access rtable for scanrelid %d", scanrelid);
        return columns;
    }

    // Get RangeTblEntry
    if (scanrelid > list_length(currentPlannedStmt->rtable)) {
        PGX_WARNING("scanrelid exceeds rtable length");
        return columns;
    }

    RangeTblEntry* rte =
        static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, scanrelid - POSTGRESQL_VARNO_OFFSET));

    if (!rte || rte->relid == InvalidOid) {
        PGX_WARNING("Invalid RTE for table schema discovery");
        return columns;
    }

    // Open relation to get schema information - CRITICAL: Preserve exact lock pattern
    Relation rel = table_open(rte->relid, AccessShareLock);
    if (!rel) {
        PGX_ERROR("Failed to open relation %d", rte->relid);
        return columns;
    }

    TupleDesc tupleDesc = RelationGetDescr(rel);
    if (!tupleDesc) {
        PGX_ERROR("Failed to get tuple descriptor");
        table_close(rel, AccessShareLock);
        return columns;
    }

    // Iterate through all table columns
    for (int i = FIRST_ATTRIBUTE_INDEX; i < tupleDesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupleDesc, i);
        if (attr->attisdropped) {
            continue; // Skip dropped columns
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

auto is_column_nullable(const PlannedStmt* currentPlannedStmt, int varno, const AttrNumber varattno) -> bool {
    // Default to nullable if we can't determine
    if (!currentPlannedStmt || !currentPlannedStmt->rtable || varno <= INVALID_VARNO || varattno <= INVALID_VARATTNO) {
        return true;
    }

#ifdef BUILDING_UNIT_TESTS
    // For unit tests, assume nullable
    return true;
#else
    // Get RangeTblEntry
    if (varno > list_length(currentPlannedStmt->rtable)) {
        return true;
    }

    RangeTblEntry* rte =
        static_cast<RangeTblEntry*>(list_nth(currentPlannedStmt->rtable, varno - POSTGRESQL_VARNO_OFFSET));
    if (!rte || rte->relid == InvalidOid) {
        return true;
    }

    // Open relation to check column nullability
    Relation rel = table_open(rte->relid, AccessShareLock);
    if (!rel) {
        return true;
    }

    TupleDesc tupleDesc = RelationGetDescr(rel);
    if (!tupleDesc) {
        table_close(rel, AccessShareLock);
        return true;
    }

    // varattno is 1-based, array is 0-based
    int attrIndex = varattno - 1;
    if (attrIndex < 0 || attrIndex >= tupleDesc->natts) {
        table_close(rel, AccessShareLock);
        return true;
    }

    Form_pg_attribute attr = TupleDescAttr(tupleDesc, attrIndex);
    bool nullable = !attr->attnotnull;

    table_close(rel, AccessShareLock);
    return nullable;
#endif
}

} // namespace postgresql_ast