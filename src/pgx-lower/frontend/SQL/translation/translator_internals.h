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
struct MergeJoin;
struct HashJoin;
struct NestLoop;
struct Material;

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/utility/logging.h"

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
#include <utility>

// ===========================================================================
// Translation Context
// ===========================================================================

namespace pgx_lower::frontend::sql {

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

using ColumnMapping = std::map<std::pair<int, int>, std::pair<std::string, std::string>>;

struct TranslationContext {
    const PlannedStmt current_stmt;
    mlir::OpBuilder& builder;
    const mlir::ModuleOp current_module;
    const mlir::Value current_tuple;

    TranslationContext(const PlannedStmt& current_stmt, mlir::OpBuilder& builder, const mlir::ModuleOp& current_module,
                       const mlir::Value& current_tuple, const ColumnMapping& mappings)
    : current_stmt(current_stmt)
    , builder(builder)
    , current_module(current_module)
    , current_tuple(current_tuple)
    , column_mappings(mappings) {}

    void set_column_mapping(int varno, int varattno, const std::string& table_name, const std::string& column_name) {
        const auto key = std::make_pair(varno, varattno);
        const auto value = std::make_pair(table_name, column_name);
        const auto it = column_mappings.find(key);
        if (it != column_mappings.end()) {
            if (it->second != value) {
                PGX_ERROR("Attempted to overwrite column mapping for varno=%d, varattno=%d: "
                          "existing=(%s, %s), new=(%s, %s)",
                          varno, varattno, it->second.first.c_str(), it->second.second.c_str(), table_name.c_str(),
                          column_name.c_str());
                throw std::runtime_error("Column mapping conflict - attempted to overwrite with different value");
            }
            return;
        }
        column_mappings[key] = value;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Set column mapping: varno=%d, varattno=%d -> (%s, %s)", varno, varattno,
                table_name.c_str(), column_name.c_str());
    }

    std::optional<std::pair<std::string, std::string>> get_column_mapping(int varno, int varattno) const {
        const auto key = std::make_pair(varno, varattno);
        const auto it = column_mappings.find(key);
        if (it != column_mappings.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Get a copy of all column mappings for context propagation
    ColumnMapping get_all_column_mappings() const { return column_mappings; }

   private:
    ColumnMapping column_mappings;
};

struct TranslationResult {
    mlir::Operation* op = nullptr;

    struct ColumnSchema {
        std::string table_name;
        std::string column_name;
        Oid type_oid;
        int32_t typmod;
        mlir::Type mlir_type;
        bool nullable;

        std::string toString() const {
            return "ColumnSchema(table='" + table_name + "', column='" + column_name
                   + "', oid=" + std::to_string(type_oid) + ", typmod=" + std::to_string(typmod)
                   + ", nullable=" + (nullable ? "true" : "false") + ")";
        }
    };

    std::vector<ColumnSchema> columns;

    std::string toString() const {
        std::string result = "TranslationResult(op=" + (op ? std::to_string(reinterpret_cast<uintptr_t>(op)) : "null")
                             + ", columns=[";
        for (size_t i = 0; i < columns.size(); ++i) {
            result += "\n\t" + columns[i].toString();
        }
        result += "])";
        return result;
    }
};

struct StreamExpressionResult {
    mlir::Value stream;
    mlir::relalg::ColumnRefAttr column_ref;
    std::string column_name;
    std::string table_name;
};

} // namespace pgx_lower::frontend::sql

namespace postgresql_ast {

using QueryCtxT = pgx_lower::frontend::sql::TranslationContext;
using TranslationResult = pgx_lower::frontend::sql::TranslationResult;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
// ===========================================================================
class PostgreSQLASTTranslator::Impl {
   public:
    explicit Impl(mlir::MLIRContext& context)
    : context_(context) {}

    auto translate_query(const PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp>;

    mlir::Value translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr);
    auto translate_expression(const QueryCtxT& ctx, Expr* expr) -> mlir::Value;
    auto translate_expression_for_stream(const QueryCtxT& ctx, Expr* expr, mlir::Value input_stream,
                                         const std::string& suggested_name,
                                         const std::vector<TranslationResult::ColumnSchema>& child_columns)
        -> pgx_lower::frontend::sql::StreamExpressionResult;
    auto translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr) -> mlir::Value;
    auto translate_var(const QueryCtxT& ctx, const Var* var) const -> mlir::Value;
    auto translate_const(const QueryCtxT& ctx, Const* const_node) const -> mlir::Value;
    auto translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr) -> mlir::Value;
    auto translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr) -> mlir::Value;
    auto translate_null_test(const QueryCtxT& ctx, const NullTest* null_test) -> mlir::Value;
    auto translate_aggref(const QueryCtxT& ctx, const Aggref* aggref) const -> mlir::Value;
    auto translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr) -> mlir::Value;
    auto translate_scalar_array_op_expr(const QueryCtxT& ctx, const ScalarArrayOpExpr* scalar_array_op) -> mlir::Value;
    auto translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr) -> mlir::Value;
    auto translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr, mlir::Value case_test_value)
        -> mlir::Value;

    // Plan node translation methods
    auto translate_plan_node(QueryCtxT& ctx, Plan* plan) -> TranslationResult;
    auto translate_seq_scan(QueryCtxT& ctx, SeqScan* seqScan) const -> TranslationResult;
    auto translate_agg(QueryCtxT& ctx, const Agg* agg) -> TranslationResult;
    auto translate_sort(QueryCtxT& ctx, const Sort* sort) -> TranslationResult;
    auto translate_limit(QueryCtxT& ctx, const Limit* limit) -> TranslationResult;
    auto translate_gather(QueryCtxT& ctx, const Gather* gather) -> TranslationResult;
    auto translate_merge_join(QueryCtxT& ctx, MergeJoin* mergeJoin) -> TranslationResult;
    auto translate_hash_join(QueryCtxT& ctx, HashJoin* hashJoin) -> TranslationResult;
    auto translate_hash(QueryCtxT& ctx, Hash* hash) -> TranslationResult;
    auto translate_nest_loop(QueryCtxT& ctx, NestLoop* nestLoop) -> TranslationResult;
    auto translate_material(QueryCtxT& ctx, Material* material) -> TranslationResult;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(const PlannedStmt* planned_stmt, QueryCtxT& context) -> bool;

    // Relational operation helpers
    auto apply_selection_from_qual(const QueryCtxT& ctx, const TranslationResult& input, const List* qual)
        -> TranslationResult;
    auto apply_projection_from_target_list(const QueryCtxT& ctx, const TranslationResult& input, const List* target_list)
        -> TranslationResult;
    auto apply_projection_from_translation_result(const QueryCtxT& ctx, const TranslationResult& input,
                                                  const TranslationResult& left_child, const TranslationResult& right_child,
                                                  const List* target_list) -> TranslationResult;

    auto create_materialize_op(const QueryCtxT& context, mlir::Value tuple_stream,
                               const TranslationResult& translation_result) const -> mlir::Value;

    // Operation translation helpers
    auto extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr)
        -> std::optional<std::pair<mlir::Value, mlir::Value>>;
    static auto translate_arithmetic_op(const QueryCtxT& context, const Oid op_oid, const mlir::Value lhs,
                                        const mlir::Value rhs) -> mlir::Value;
    static auto upcast_binary_operation(const QueryCtxT& ctx, mlir::Value lhs, mlir::Value rhs)
        -> std::pair<mlir::Value, mlir::Value>;
    static auto translate_comparison_op(const QueryCtxT& context, const Oid op_oid, const mlir::Value lhs,
                                        const mlir::Value rhs) -> mlir::Value;

   private:
    mlir::MLIRContext& context_;
};

// ===========================================================================
// PostgreSQLTypeMapper
// ===========================================================================
class PostgreSQLTypeMapper {
   public:
    explicit PostgreSQLTypeMapper(mlir::MLIRContext& context)
    : context_(context) {}

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