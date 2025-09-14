#pragma once

extern "C" {
#include "postgres.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"
}

using AttrNumber = int16;
using Bitmapset = struct Bitmapset;

// fwd
struct Agg;
struct Sort;
struct Limit;
struct Gather;

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
// Translation Context
// ===========================================================================

namespace pgx_lower::frontend::sql {

// Type aliases for column mapping
using varno_t = int;
using varattno_t = int;
using table_t = std::string;
using column_t = std::string;
using ColumnMapping = std::map<std::pair<varno_t, varattno_t>, std::pair<table_t, column_t>>;

struct ColumnInfo {
    std::string name;
    unsigned int typeOid;
    int32_t typmod;
    bool nullable;

    ColumnInfo(std::string n, const unsigned int t, const int32_t m, const bool null)
    : name(std::move(n))
    , typeOid(t)
    , typmod(m)
    , nullable(null) {}
};

struct TranslationContext {
    PlannedStmt* currentStmt = nullptr;
    mlir::OpBuilder* builder = nullptr;
    mlir::Value currentTuple = nullptr;
    ColumnMapping columnMappings;
};

} // namespace pgx_lower::frontend::sql

namespace postgresql_ast {

using TranslationContextT = pgx_lower::frontend::sql::TranslationContext;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
// ===========================================================================
class PostgreSQLASTTranslator::Impl {
   public:
    explicit Impl(mlir::MLIRContext& context)
    : context_(context)
    , builder_(nullptr)
    , current_module_(nullptr)
    , current_tuple_handle_(nullptr)
    , current_planned_stmt_(nullptr)
    , context_needs_recreation_(false) {}

    auto translate_query(PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp>;

    auto translate_expression(Expr* expr) -> mlir::Value;
    auto translate_op_expr(const OpExpr* op_expr) -> mlir::Value;
    auto translate_var(const Var* var) const -> mlir::Value;
    auto translate_const(Const* const_node) const -> mlir::Value;
    auto translate_func_expr(const FuncExpr* func_expr) -> mlir::Value;
    auto translate_bool_expr(const BoolExpr* bool_expr) -> mlir::Value;
    auto translate_null_test(const NullTest* null_test) -> mlir::Value;
    auto translate_aggref(const Aggref* aggref) const -> mlir::Value;
    auto translate_coalesce_expr(const CoalesceExpr* coalesce_expr) -> mlir::Value;
    auto translate_scalar_array_op_expr(const ScalarArrayOpExpr* scalar_array_op) -> mlir::Value;
    auto translate_case_expr(const CaseExpr* case_expr) -> mlir::Value;
    auto translate_expression_with_case_test(Expr* expr, mlir::Value case_test_value) -> mlir::Value;

    // Plan node translation methods
    auto translate_plan_node(Plan* plan, TranslationContextT& context) -> mlir::Operation*;
    auto translate_seq_scan(SeqScan* seqScan, TranslationContextT& context) const -> mlir::Operation*;
    auto translate_agg(Agg* agg, TranslationContextT& context) -> mlir::Operation*;
    auto translate_sort(const Sort* sort, TranslationContextT& context) -> mlir::Operation*;
    auto translate_limit(const Limit* limit, TranslationContextT& context) -> mlir::Operation*;
    auto translate_gather(const Gather* gather, TranslationContextT& context) -> mlir::Operation*;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder, const TranslationContextT& context) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(mlir::func::FuncOp query_func,
                                     const PlannedStmt* planned_stmt,
                                     TranslationContextT& context) -> bool;

    // Relational operation helpers
    auto apply_selection_from_qual(mlir::Operation* input_op, const List* qual, const TranslationContextT& context)
        -> mlir::Operation*;
    auto apply_projection_from_target_list(mlir::Operation* input_op, List* target_list, TranslationContextT& context)
        -> mlir::Operation*;

    // Validation and column processing
    static auto validate_plan_tree(const Plan* plan_tree) -> bool;
    auto extract_target_list_columns(TranslationContextT& context,
                                     std::vector<mlir::Attribute>& column_ref_attrs,
                                     std::vector<mlir::Attribute>& column_name_attrs) const -> bool;
    auto process_target_entry(TranslationContextT& context,
                              const List* t_list,
                              int index,
                              std::vector<mlir::Attribute>& column_ref_attrs,
                              std::vector<mlir::Attribute>& column_name_attrs) const -> bool;
    auto determine_column_type(const TranslationContextT& context, Expr* expr) const -> mlir::Type;
    auto create_materialize_op(TranslationContextT& context, mlir::Value tuple_stream) const -> mlir::Operation*;

    // Operation translation helpers
    auto extract_op_expr_operands(const OpExpr* op_expr, mlir::Value& lhs, mlir::Value& rhs) -> bool;
    auto translate_arithmetic_op(Oid op_oid, mlir::Value lhs, mlir::Value rhs) const -> mlir::Value;
    auto translate_comparison_op(Oid op_oid, mlir::Value lhs, mlir::Value rhs) const -> mlir::Value;

   private:
    // Helper to map PostgreSQL aggregate function OIDs to LingoDB function names
    static auto get_aggregate_function_name(Oid aggfnoid) -> const char*;

    mlir::MLIRContext& context_;
    mlir::OpBuilder* builder_;
    mlir::ModuleOp* current_module_;
    mlir::Value* current_tuple_handle_;
    PlannedStmt* current_planned_stmt_;
    bool context_needs_recreation_;
};

// ===========================================================================
// PostgreSQLTypeMapper
// ===========================================================================
class PostgreSQLTypeMapper {
   public:
    explicit PostgreSQLTypeMapper(mlir::MLIRContext& context)
    : context_(context) {}

    // Main type mapping function
    auto map_postgre_sqltype(Oid type_oid, int32_t typmod, bool nullable = false) const -> mlir::Type;

    // Type modifier extraction functions
    static auto extract_numeric_info(int32_t typmod) -> std::pair<int32_t, int32_t>;
    static auto extract_timestamp_precision(int32_t typmod) -> mlir::db::TimeUnitAttr;
    static auto extract_varchar_length(int32_t typmod) -> int32_t;

   private:
    mlir::MLIRContext& context_;
};

// ===========================================================================
// SchemaManager
// ===========================================================================
class SchemaManager {
   public:
    explicit SchemaManager(PlannedStmt* planned_stmt)
    : planned_stmt_(planned_stmt) {}

    // Table and column metadata access
    auto get_table_name_from_rte(int varno) -> std::string;
    auto get_column_name_from_schema(int rt_index, AttrNumber attnum) -> std::string;
    auto get_table_oid_from_rte(int varno) -> Oid;
    auto is_column_nullable(int rtindex, AttrNumber attnum) -> bool;

   private:
    PlannedStmt* planned_stmt_;
};

// ===========================================================================
// Helper Functions
// ===========================================================================

// Aggregate function name mapping
auto get_aggregate_function_name(Oid aggfnoid) -> const char*;

// Schema access helpers (standalone functions for backwards compatibility)
auto get_table_name_from_rte(PlannedStmt* current_planned_stmt, int varno) -> std::string;
auto get_column_name_from_schema(const PlannedStmt* planned_stmt, int rt_index, AttrNumber attnum) -> std::string;
auto get_table_oid_from_rte(PlannedStmt* current_planned_stmt, int varno) -> Oid;
auto is_column_nullable(const PlannedStmt* planned_stmt, int rt_index, AttrNumber attnum) -> bool;

// Translation helper for constants
auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;

// Get all columns from a table
auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast