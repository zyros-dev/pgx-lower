#pragma once

extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"

typedef int16 AttrNumber;
typedef struct Bitmapset Bitmapset;

typedef struct Agg Agg;
typedef struct Sort Sort;
typedef struct Limit Limit;
typedef struct Gather Gather;

#define AGG_PLAIN 0
#define AGG_SORTED 1
#define AGG_HASHED 2
#define AGG_MIXED 3
}

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Translation Context Types
// ===========================================================================

namespace pgx_lower::frontend::sql {

// Type aliases for column mapping clarity
using varno_t = int;
using varattno_t = int;
using table_t = std::string;
using column_t = std::string;
using ColumnMapping = std::map<std::pair<varno_t, varattno_t>, std::pair<table_t, column_t>>;

// Column information structure for schema discovery
struct ColumnInfo {
    std::string name;
    unsigned int typeOid; // PostgreSQL type OID
    int32_t typmod; // Type modifier
    bool nullable;

    ColumnInfo(std::string n, const unsigned int t, const int32_t m, const bool null)
    : name(std::move(n))
    , typeOid(t)
    , typmod(m)
    , nullable(null) {}
};

// Translation context for managing state during AST translation
struct TranslationContext {
    PlannedStmt* currentStmt = nullptr;
    mlir::OpBuilder* builder = nullptr;
    std::unordered_map<unsigned int, mlir::Type> typeCache; // Oid -> Type mapping
    mlir::Value currentTuple = nullptr;
    ColumnMapping columnMappings; // Maps (varno, varattno) -> (table_name, column_name)
};

} // namespace pgx_lower::frontend::sql

namespace postgresql_ast {

using TranslationContext = pgx_lower::frontend::sql::TranslationContext;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
// ===========================================================================
class PostgreSQLASTTranslator::Impl {
   public:
    explicit Impl(mlir::MLIRContext& context)
    : context_(context)
    , builder_(nullptr)
    , currentModule_(nullptr)
    , currentTupleHandle_(nullptr)
    , currentPlannedStmt_(nullptr)
    , contextNeedsRecreation_(false) {}

    auto translate_query(PlannedStmt* plannedStmt) -> std::unique_ptr<mlir::ModuleOp>;

    auto translate_expression(Expr* expr) -> mlir::Value;
    auto translate_op_expr(const OpExpr* opExpr) -> mlir::Value;
    auto translate_var(const Var* var) const -> mlir::Value;
    auto translate_const(Const* constNode) const -> mlir::Value;
    auto translate_func_expr(const FuncExpr* funcExpr) -> mlir::Value;
    auto translate_bool_expr(const BoolExpr* boolExpr) -> mlir::Value;
    auto translate_null_test(const NullTest* nullTest) -> mlir::Value;
    auto translate_aggref(const Aggref* aggref) const -> mlir::Value;
    auto translate_coalesce_expr(const CoalesceExpr* coalesceExpr) -> mlir::Value;
    auto translate_scalar_array_op_expr(const ScalarArrayOpExpr* scalarArrayOp) -> mlir::Value;
    auto translate_case_expr(const CaseExpr* caseExpr) -> mlir::Value;
    auto translate_expression_with_case_test(Expr* expr, mlir::Value caseTestValue) -> mlir::Value;

    // Plan node translation methods
    auto translate_plan_node(Plan* plan, TranslationContext& context) -> mlir::Operation*;
    auto translate_seq_scan(SeqScan* seqScan, TranslationContext& context) const -> mlir::Operation*;
    auto translate_agg(Agg* agg, TranslationContext& context) -> mlir::Operation*;
    auto translate_sort(const Sort* sort, TranslationContext& context) -> mlir::Operation*;
    auto translate_limit(const Limit* limit, TranslationContext& context) -> mlir::Operation*;
    auto translate_gather(const Gather* gather, TranslationContext& context) -> mlir::Operation*;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder, const TranslationContext& context) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(mlir::func::FuncOp queryFunc,
                                     const PlannedStmt* plannedStmt,
                                     TranslationContext& context) -> bool;

    // Relational operation helpers
    auto apply_selection_from_qual(mlir::Operation* inputOp, const List* qual, const TranslationContext& context)
        -> mlir::Operation*;
    auto apply_projection_from_target_list(mlir::Operation* inputOp, List* targetList, TranslationContext& context)
        -> mlir::Operation*;

    // Validation and column processing
    static auto validate_plan_tree(const Plan* planTree) -> bool;
    auto extract_target_list_columns(TranslationContext& context,
                                     std::vector<mlir::Attribute>& columnRefAttrs,
                                     std::vector<mlir::Attribute>& columnNameAttrs) const -> bool;
    auto process_target_entry(TranslationContext& context,
                              const List* tlist,
                              int index,
                              std::vector<mlir::Attribute>& columnRefAttrs,
                              std::vector<mlir::Attribute>& columnNameAttrs) const -> bool;
    auto determine_column_type(const TranslationContext& context, Expr* expr) const -> mlir::Type;
    auto create_materialize_op(TranslationContext& context, mlir::Value tupleStream) const -> mlir::Operation*;

    // Operation translation helpers
    auto extract_op_expr_operands(const OpExpr* opExpr, mlir::Value& lhs, mlir::Value& rhs) -> bool;
    auto translate_arithmetic_op(Oid opOid, mlir::Value lhs, mlir::Value rhs) const -> mlir::Value;
    auto translate_comparison_op(Oid opOid, mlir::Value lhs, mlir::Value rhs) const -> mlir::Value;

   private:
    // Helper to map PostgreSQL aggregate function OIDs to LingoDB function names
    static auto get_aggregate_function_name(Oid aggfnoid) -> const char*;

    mlir::MLIRContext& context_;
    mlir::OpBuilder* builder_;
    mlir::ModuleOp* currentModule_;
    mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;
};

// ===========================================================================
// PostgreSQLTypeMapper
// ===========================================================================
class PostgreSQLTypeMapper {
   public:
    explicit PostgreSQLTypeMapper(mlir::MLIRContext& context)
    : context_(context) {}

    // Main type mapping function
    auto map_postgre_sqltype(Oid typeOid, int32_t typmod, bool nullable = false) -> mlir::Type;

    // Type modifier extraction functions
    auto extract_numeric_info(int32_t typmod) -> std::pair<int32_t, int32_t>;
    auto extract_timestamp_precision(int32_t typmod) -> mlir::db::TimeUnitAttr;
    auto extract_varchar_length(int32_t typmod) -> int32_t;

   private:
    mlir::MLIRContext& context_;
};

// ===========================================================================
// SchemaManager
// ===========================================================================
class SchemaManager {
   public:
    explicit SchemaManager(PlannedStmt* plannedStmt)
    : plannedStmt_(plannedStmt) {}

    // Table and column metadata access
    auto get_table_name_from_rte(int varno) -> std::string;
    auto get_column_name_from_schema(int rtindex, AttrNumber attnum) -> std::string;
    auto get_table_oid_from_rte(int varno) -> Oid;
    auto is_column_nullable(int rtindex, AttrNumber attnum) -> bool;

   private:
    PlannedStmt* plannedStmt_;
};

// ===========================================================================
// Helper Functions
// ===========================================================================

// Aggregate function name mapping
auto get_aggregate_function_name(Oid aggfnoid) -> const char*;

// Schema access helpers (standalone functions for backwards compatibility)
auto get_table_name_from_rte(PlannedStmt* currentPlannedStmt, int varno) -> std::string;
auto get_column_name_from_schema(const PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> std::string;
auto get_table_oid_from_rte(PlannedStmt* currentPlannedStmt, int varno) -> Oid;
auto is_column_nullable(const PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> bool;

// Translation helper for constants
auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;

// Get all columns from a table
auto get_all_table_columns_from_schema(const PlannedStmt* currentPlannedStmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast