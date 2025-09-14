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
#include <map>
#include <string>
#include <vector>
#include <optional>

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
    unsigned int type_oid;
    int32_t typmod;
    bool nullable;

    ColumnInfo(std::string n, const unsigned int t, const int32_t m, const bool null)
    : name(std::move(n))
    , type_oid(t)
    , typmod(m)
    , nullable(null) {}
};

struct TranslationContext {
    const PlannedStmt current_stmt;
    mlir::OpBuilder& builder;
    const mlir::ModuleOp current_module;
    const mlir::Value current_tuple;

    ColumnMapping column_mappings;
};

} // namespace pgx_lower::frontend::sql

namespace postgresql_ast {

using QueryCtxT = pgx_lower::frontend::sql::TranslationContext;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
// ===========================================================================
class PostgreSQLASTTranslator::Impl {
   public:
    explicit Impl(mlir::MLIRContext& context)
    : context_(context) {}

    auto translate_query(const PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp>;

    auto translate_expression(const QueryCtxT& ctx, Expr* expr) -> mlir::Value;
    auto translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr) -> mlir::Value;
    auto translate_var(const QueryCtxT& ctx, const Var* var) const -> mlir::Value;
    auto translate_const(const QueryCtxT& ctx, Const* const_node) const -> mlir::Value;
    auto translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr) -> mlir::Value;
    auto translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr) -> mlir::Value;
    auto translate_null_test(const QueryCtxT& ctx, const NullTest* null_test) -> mlir::Value;
    static auto translate_aggref(const QueryCtxT& ctx, const Aggref* aggref) -> mlir::Value;
    auto translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr) -> mlir::Value;
    auto translate_scalar_array_op_expr(const QueryCtxT& ctx, const ScalarArrayOpExpr* scalar_array_op) -> mlir::Value;
    auto translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr) -> mlir::Value;
    auto translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr, mlir::Value case_test_value)
        -> mlir::Value;

    // Plan node translation methods
    auto translate_plan_node(QueryCtxT& ctx, Plan* plan) -> mlir::Operation*;
    auto translate_seq_scan(QueryCtxT& ctx, SeqScan* seqScan) const -> mlir::Operation*;
    auto translate_agg(QueryCtxT& ctx, const Agg* agg) -> mlir::Operation*;
    auto translate_sort(QueryCtxT& ctx, const Sort* sort) -> mlir::Operation*;
    auto translate_limit(QueryCtxT& ctx, const Limit* limit) -> mlir::Operation*;
    auto translate_gather(QueryCtxT& ctx, const Gather* gather) -> mlir::Operation*;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder, const QueryCtxT& context) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(const mlir::func::FuncOp query_func, const PlannedStmt* planned_stmt,
                                     QueryCtxT& context) -> bool;

    // Relational operation helpers
    auto apply_selection_from_qual(const QueryCtxT& ctx, mlir::Operation* input_op, const List* qual) -> mlir::Operation*;
    auto apply_projection_from_target_list(const QueryCtxT& ctx, mlir::Operation* input_op, const List* target_list)
        -> mlir::Operation*;

    // Validation and column processing
    static auto validate_plan_tree(const Plan* plan_tree) -> bool;
    auto extract_target_list_columns(const QueryCtxT& context, std::vector<mlir::Attribute>& column_ref_attrs,
                                     std::vector<mlir::Attribute>& column_name_attrs) const -> bool;
    auto process_target_entry(const QueryCtxT& context, const List* t_list, int index,
                              std::vector<mlir::Attribute>& column_ref_attrs,
                              std::vector<mlir::Attribute>& column_name_attrs) const -> bool;
    auto determine_column_type(const QueryCtxT& context, Expr* expr) const -> mlir::Type;
    auto create_materialize_op(const QueryCtxT& context, mlir::Value tuple_stream) const -> mlir::Operation*;

    // Operation translation helpers
    auto extract_op_expr_operands(const QueryCtxT& context, const OpExpr* op_expr)
        -> std::optional<std::pair<mlir::Value, mlir::Value>>;
    static auto translate_arithmetic_op(const QueryCtxT& context, const Oid op_oid, const mlir::Value lhs,
                                        const mlir::Value rhs) -> mlir::Value;
    static auto translate_comparison_op(const QueryCtxT& context, const Oid op_oid, const mlir::Value lhs,
                                        const mlir::Value rhs) -> mlir::Value;

   private:
    static auto get_aggregate_function_name(const Oid aggfnoid) -> const char*;

    mlir::MLIRContext& context_;
};

// ===========================================================================
// PostgreSQLTypeMapper
// ===========================================================================
class PostgreSQLTypeMapper {
   public:
    explicit PostgreSQLTypeMapper(mlir::MLIRContext& context)
    : context_(context) {}

    // Main type mapping function
    auto map_postgre_sqltype(const Oid type_oid, const int32_t typmod, const bool nullable = false) const -> mlir::Type;

    // Type modifier extraction functions
    static auto extract_numeric_info(const int32_t typmod) -> std::pair<int32_t, int32_t>;
    static auto extract_timestamp_precision(const int32_t typmod) -> mlir::db::TimeUnitAttr;
    static auto extract_varchar_length(const int32_t typmod) -> int32_t;

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
    auto get_table_name_from_rte(const int varno) -> std::string;
    auto get_column_name_from_schema(const int rt_index, AttrNumber attnum) -> std::string;
    auto get_table_oid_from_rte(const int varno) -> Oid;
    auto is_column_nullable(const int rtindex, const AttrNumber attnum) -> bool;

   private:
    PlannedStmt* planned_stmt_;
};

// ===========================================================================
// Helper Functions
// ===========================================================================

// Aggregate function name mapping
auto get_aggregate_function_name(Oid aggfnoid) -> const char*;

// Schema access helpers (standalone functions for backwards compatibility)
auto get_table_name_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> std::string;
auto get_column_name_from_schema(const PlannedStmt* planned_stmt, const int rt_index, AttrNumber attnum) -> std::string;
auto get_table_oid_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> Oid;
auto is_column_nullable(const PlannedStmt* planned_stmt, const int rt_index, const AttrNumber attnum) -> bool;

// Translation helper for constants
auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;

// Get all columns from a table
auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, const int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast