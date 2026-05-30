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

auto get_table_name_from_rte(const PlannedStmt* current_planned_stmt, const int VARNO) -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!current_planned_stmt || !current_planned_stmt->rtable || VARNO <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, VARNO);
        throw std::runtime_error("Invalid RTE");
    }

    if (VARNO > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", VARNO, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid RTE");
    }

    auto *const RTE = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, VARNO - POSTGRESQL_VARNO_OFFSET));

    if (!RTE || RTE->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for varno %d", VARNO);
        throw std::runtime_error("Invalid RTE");
    }

#ifdef BUILDING_UNIT_TESTS
    return std::string(UNIT_TEST_TABLE_PREFIX) + std::to_string(varno);
#else
    char* const relname = get_rel_name(RTE->relid);
    std::string table_name = relname ? relname : ("unknown_table_" + std::to_string(VARNO));

    return table_name;
#endif
}

auto get_table_alias_from_rte(const PlannedStmt* current_planned_stmt, const int VARNO) -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!current_planned_stmt || !current_planned_stmt->rtable || VARNO <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, VARNO);
        throw std::runtime_error("Invalid RTE");
    }

    if (VARNO > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", VARNO, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid RTE");
    }

    auto *const RTE = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, VARNO - POSTGRESQL_VARNO_OFFSET));

    if (!RTE) {
        PGX_ERROR("Invalid RTE for varno %d", VARNO);
        throw std::runtime_error("Invalid RTE");
    }

#ifdef BUILDING_UNIT_TESTS
    return std::string(UNIT_TEST_TABLE_PREFIX) + std::to_string(varno);
#else
    if (RTE->eref && RTE->eref->aliasname) {
        return std::string(RTE->eref->aliasname);
    }

    if (RTE->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for varno %d", VARNO);
        throw std::runtime_error("Invalid RTE");
    }

    char* const relname = get_rel_name(RTE->relid);
    return relname ? relname : ("unknown_table_" + std::to_string(VARNO));
#endif
}

auto get_column_name_from_schema(const PlannedStmt* current_planned_stmt, const int VARNO, const AttrNumber VARATTNO)
    -> std::string {
    PGX_IO(AST_TRANSLATE);
    if (!current_planned_stmt || !current_planned_stmt->rtable || VARNO <= INVALID_VARNO || VARATTNO <= INVALID_VARATTNO) {
        PGX_ERROR("Cannot access schema for column: varno=%d varattno=%d", VARNO, VARATTNO);
        throw std::runtime_error("Invalid - read logs");
    }

    if (VARNO > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno exceeds rtable length");
        throw std::runtime_error("Invalid - read logs");
    }

    auto *const RTE = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, VARNO - POSTGRESQL_VARNO_OFFSET));

    if (!RTE) {
        PGX_ERROR("Invalid RTE for column lookup: varno=%d", VARNO);
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
    if (RTE->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for column lookup: varno=%d has no relid (CTE/subquery should use varno_resolution)",
                  VARNO);
        throw std::runtime_error("Invalid - read logs");
    }

    char* const attname = get_attname(RTE->relid, VARATTNO, PG_ATTNAME_NOT_MISSING_OK);
    std::string column_name = attname ? attname : ("col_" + std::to_string(VARATTNO));

    return column_name;
#endif
}

auto get_table_oid_from_rte(const PlannedStmt* current_planned_stmt, const int VARNO) -> Oid {
    PGX_IO(AST_TRANSLATE);
    using namespace pgx_lower::frontend::sql::constants;
    if (!current_planned_stmt || !current_planned_stmt->rtable || VARNO <= INVALID_VARNO) {
        PGX_ERROR("Cannot access rtable: currentPlannedStmt=%p varno=%d", current_planned_stmt, VARNO);
        throw std::runtime_error("Invalid - read logs");
    }

    if (VARNO > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("varno %d exceeds rtable length %d", VARNO, list_length(current_planned_stmt->rtable));
        throw std::runtime_error("Invalid - read logs");
    }

    auto *const RTE = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, VARNO - POSTGRESQL_VARNO_OFFSET));

    if (!RTE) {
        PGX_ERROR("Invalid RTE for varno %d", VARNO);
        throw std::runtime_error("Invalid - read logs");
    }

    return RTE->relid;
}

auto is_column_nullable(const PlannedStmt* current_planned_stmt, const int VARNO, const AttrNumber VARATTNO) -> bool {
    PGX_IO(AST_TRANSLATE);

    if (!current_planned_stmt || !current_planned_stmt->rtable || VARNO <= INVALID_VARNO || VARATTNO <= INVALID_VARATTNO) {
        return true;
    }

#ifdef BUILDING_UNIT_TESTS
    return true;
#else
    if (VARNO > list_length(current_planned_stmt->rtable)) {
        return true;
    }

    auto *const RTE = static_cast<RangeTblEntry*>(list_nth(current_planned_stmt->rtable, VARNO - POSTGRESQL_VARNO_OFFSET));
    if (!RTE || RTE->relid == InvalidOid) {
        return true;
    }

    auto *const REL = table_open(RTE->relid, AccessShareLock);
    if (!REL) {
        return true;
    }

    auto *const TUPLE_DESC = RelationGetDescr(REL);
    if (!TUPLE_DESC) {
        table_close(REL, AccessShareLock);
        return true;
    }

    const int ATTR_INDEX = VARATTNO - 1;
    if (ATTR_INDEX < 0 || ATTR_INDEX >= TUPLE_DESC->natts) {
        table_close(REL, AccessShareLock);
        return true;
    }

    const Form_pg_attribute ATTR = TupleDescAttr(TUPLE_DESC, ATTR_INDEX);
    const bool NULLABLE = !ATTR->attnotnull;

    table_close(REL, AccessShareLock);
    return NULLABLE;
#endif
}

auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, const int SCANRELID)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo> {
    PGX_IO(AST_TRANSLATE);
    std::vector<pgx_lower::frontend::sql::ColumnInfo> columns;

#ifdef BUILDING_UNIT_TESTS
    columns.emplace_back("id", INT4OID, INVALID_TYPMOD, UNIT_TEST_COLUMN_NOT_NULL);
    return columns;
#else
    if (!current_planned_stmt || !current_planned_stmt->rtable || SCANRELID <= 0) {
        PGX_ERROR("Cannot access rtable for scanrelid %d", SCANRELID);
        throw std::runtime_error("Invalid - read logs");
    }

    if (SCANRELID > list_length(current_planned_stmt->rtable)) {
        PGX_ERROR("scanrelid exceeds rtable length");
        throw std::runtime_error("Invalid - read logs");
    }

    auto *const RTE = static_cast<RangeTblEntry*>(
        list_nth(current_planned_stmt->rtable, SCANRELID - POSTGRESQL_VARNO_OFFSET));

    if (!RTE || RTE->relid == InvalidOid) {
        PGX_ERROR("Invalid RTE for table schema discovery");
        throw std::runtime_error("Invalid - read logs");
    }

    const Relation REL = table_open(RTE->relid, AccessShareLock);
    if (!REL) {
        PGX_ERROR("Failed to open relation %d", RTE->relid);
        throw std::runtime_error("Invalid - read logs");
    }

    const TupleDesc TUPLE_DESC = RelationGetDescr(REL);
    if (!TUPLE_DESC) {
        PGX_ERROR("Failed to get tuple descriptor");
        table_close(REL, AccessShareLock);
        throw std::runtime_error("Invalid - read logs");
    }

    for (int i = 0; i < TUPLE_DESC->natts; i++) {
        const Form_pg_attribute ATTR = TupleDescAttr(TUPLE_DESC, i);
        if (ATTR->attisdropped) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Skipping attr");
            continue;
        }

        std::string const col_name = NameStr(ATTR->attname);
        Oid const col_type = ATTR->atttypid;
        int32_t const typmod = ATTR->atttypmod;
        bool const nullable = !ATTR->attnotnull;

        columns.emplace_back(col_name, col_type, typmod, nullable);
    }

    table_close(REL, AccessShareLock);

    PGX_LOG(AST_TRANSLATE, DEBUG, "Discovered %zu columns for scanrelid %d", columns.size(), SCANRELID);
    return columns;
#endif
}

} // namespace postgresql_ast