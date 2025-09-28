#pragma once

extern "C" {
#include "postgres.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"
}

using AttrNumber = int16;

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
#include <optional>
#include <functional>

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

template<typename T>
using OptRefT = std::optional<std::reference_wrapper<T>>;

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


struct TranslationResult {
    mlir::Operation* op = nullptr;

    struct ColumnSchema {
        std::string table_name;
        std::string column_name;
        Oid type_oid;
        int32_t typmod;
        mlir::Type mlir_type;
        bool nullable;
        auto toString() const -> std::string {
            return "ColumnSchema(table='" + table_name + "', column='" + column_name
                   + "', oid=" + std::to_string(type_oid) + ", typmod=" + std::to_string(typmod)
                   + ", nullable=" + (nullable ? "true" : "false") + ")";
        }
    };

    std::vector<ColumnSchema> columns;
    std::string current_scope;
    std::map<std::pair<int, int>, std::pair<std::string, std::string>> varno_resolution;

    [[nodiscard]] auto resolve_var(int varno, int varattno) const -> std::optional<std::pair<std::string, std::string>> {
        const auto key = std::make_pair(varno, varattno);
        const auto it = varno_resolution.find(key);
        if (it != varno_resolution.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto toString() const -> std::string {
        std::string result = "TranslationResult(op=" + (op ? std::to_string(reinterpret_cast<uintptr_t>(op)) : "null")
                             + ", scope=" + current_scope + ", columns=[";
        for (size_t i = 0; i < columns.size(); ++i) {
            result += "\n\t" + columns[i].toString();
        }
        result += "]";

        if (!varno_resolution.empty()) {
            result += ", varno_mappings=[";
            for (const auto& [key, value] : varno_resolution) {
                result += "\n\t(" + std::to_string(key.first) + "," + std::to_string(key.second) + ") -> ("
                          + value.first + ", " + value.second + ")";
            }
            result += "]";
        }

        result += ")";
        return result;
    }
};

/**
 * This class aims to contain IMMUTABLE state except for the OpBuilder. Since the OpBuilder is effectively our primary
 * goal in this crawl, its ok to constantly mutate it. However, everything else here should be immutable.
 * I'm aiming to remove the column_mappings over time -
 */
struct SubqueryInfo {
    std::string join_scope;
    std::string join_column_name;
    mlir::Type output_type;

    SubqueryInfo() = default;
};

struct TranslationContext {
    const PlannedStmt current_stmt;
    mlir::OpBuilder& builder;
    const mlir::ModuleOp current_module;
    const mlir::Value current_tuple;
    const mlir::Value outer_tuple;
    std::unordered_map<int, TranslationResult> init_plan_results;
    std::unordered_map<int, SubqueryInfo> subquery_param_mapping;
    std::unordered_map<int, std::pair<std::string, std::string>> correlation_params;
    std::unordered_map<int, Var*> nest_params;
    static int outer_join_counter;
};

inline int TranslationContext::outer_join_counter = 0;

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
template<typename T>
using OptRefT = pgx_lower::frontend::sql::OptRefT<T>;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
// ===========================================================================
class PostgreSQLASTTranslator::Impl {
   public:
    explicit Impl(mlir::MLIRContext& context)
    : context_(context) {}

    auto translate_query(const PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp>;

    mlir::Value translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr, OptRefT<const TranslationResult> current_result = std::nullopt);
    auto translate_expression(const QueryCtxT& ctx, Expr* expr,
                              OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_expression_with_join_context(const QueryCtxT& ctx, Expr* expr, const TranslationResult* left_child,
                                                const TranslationResult* right_child,
                                                std::optional<mlir::Value> outer_tuple_arg = std::nullopt) -> mlir::Value;
    auto translate_op_expr_with_join_context(const QueryCtxT& ctx, const OpExpr* op_expr,
                                             const TranslationResult* left_child, const TranslationResult* right_child,
                                             std::optional<mlir::Value> outer_tuple_arg = std::nullopt)
        -> mlir::Value;
    auto translate_bool_expr_with_join_context(const QueryCtxT& ctx, const BoolExpr* bool_expr,
                                               const TranslationResult* left_child,
                                               const TranslationResult* right_child,
                                               std::optional<mlir::Value> outer_tuple_arg = std::nullopt) -> mlir::Value;
    auto translate_func_expr_with_join_context(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                               const TranslationResult* left_child,
                                               const TranslationResult* right_child,
                                               std::optional<mlir::Value> outer_tuple_arg = std::nullopt) -> mlir::Value;
    auto translate_expression_for_stream(const QueryCtxT& ctx, Expr* expr, mlir::Value input_stream,
                                         const std::string& suggested_name,
                                         const std::vector<TranslationResult::ColumnSchema>& child_columns)
        -> pgx_lower::frontend::sql::StreamExpressionResult;

    [[deprecated("Use translate_op_expr_with_join_context() for join contexts")]]
    auto translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr,
                           OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;

    auto translate_var(const QueryCtxT& ctx, const Var* var,
                       OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_const(const QueryCtxT& ctx, Const* const_node,
                         OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr,
                             OptRefT<const TranslationResult> current_result = std::nullopt,
                             std::optional<std::vector<mlir::Value>> pre_translated_args = std::nullopt) -> mlir::Value;

    [[deprecated("Use translate_bool_expr_with_join_context() for join contexts")]]
    auto translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr,
                             OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_null_test(const QueryCtxT& ctx, const NullTest* null_test,
                             OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_aggref(const QueryCtxT& ctx, const Aggref* aggref,
                          OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr, OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_scalar_array_op_expr(const QueryCtxT& ctx, const ScalarArrayOpExpr* scalar_array_op,
                                        OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr, OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr, mlir::Value case_test_value)
        -> mlir::Value;

    // Subquery translation
    auto translate_subplan(const QueryCtxT& ctx, const SubPlan* subplan,
                          OptRefT<const TranslationResult> current_result) -> mlir::Value;
    auto translate_subquery_plan(const QueryCtxT& parent_ctx, Plan* subquery_plan,
                                 const PlannedStmt* parent_stmt)
        -> std::pair<mlir::Value, TranslationResult>;
    auto translate_param(const QueryCtxT& ctx, const Param* param,
                        OptRefT<const TranslationResult> current_result) -> mlir::Value;

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
    auto translate_subquery_scan(QueryCtxT& ctx, SubqueryScan* subqueryScan) -> TranslationResult;
    auto translate_cte_scan(QueryCtxT& ctx, CteScan* cteScan) -> TranslationResult;

    // InitPlan helpers
    auto process_init_plans(QueryCtxT& ctx, Plan* plan) -> void;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(const PlannedStmt* planned_stmt, QueryCtxT& context) -> bool;

    // Relational operation helpers

    [[deprecated("Use apply_selection_from_qual_with_columns() for better join support")]]
    auto apply_selection_from_qual(const QueryCtxT& ctx, const TranslationResult& input, const List* qual)
        -> TranslationResult;

    auto apply_selection_from_qual_with_columns(const QueryCtxT& ctx, const TranslationResult& input, const List* qual,
                                                const TranslationResult* left_child,
                                                const TranslationResult* right_child) -> TranslationResult;
    auto apply_projection_from_target_list(const QueryCtxT& ctx, const TranslationResult& input,
                                           const List* target_list, const TranslationResult* left_child = nullptr,
                                           const TranslationResult* right_child = nullptr) -> TranslationResult;
    auto apply_projection_from_translation_result(const QueryCtxT& ctx, const TranslationResult& input,
                                                  const TranslationResult& left_child,
                                                  const TranslationResult& right_child, const List* target_list)
        -> TranslationResult;

    auto create_materialize_op(const QueryCtxT& context, mlir::Value tuple_stream,
                               const TranslationResult& translation_result) const -> mlir::Value;

    // Join helper functions
    void translate_join_predicate_to_region(const QueryCtxT& ctx, mlir::Block* predicateBlock, mlir::Value tupleArg,
                                            List* joinClauses, const TranslationResult& leftTranslation,
                                            const TranslationResult& rightTranslation);

    auto create_join_operation(QueryCtxT& ctx, JoinType join_type, mlir::Value left_value, mlir::Value right_value,
                               const TranslationResult& left_translation, const TranslationResult& right_translation,
                               List* join_clauses) -> TranslationResult;

    // Operation translation helpers
    auto extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr,
                                  OptRefT<const TranslationResult> current_result = std::nullopt)
        -> std::optional<std::pair<mlir::Value, mlir::Value>>;
    static auto normalize_bpchar_operands(const QueryCtxT& ctx, const OpExpr* op_expr, mlir::Value lhs,
                                          mlir::Value rhs) -> std::pair<mlir::Value, mlir::Value>;
    static auto translate_arithmetic_op(const QueryCtxT& context, const OpExpr* op_expr, const mlir::Value lhs,
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
// Helper Functions
// ===========================================================================

// Schema access helpers (standalone functions for backwards compatibility)
auto get_table_name_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> std::string;
auto get_table_alias_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> std::string;
auto get_column_name_from_schema(const PlannedStmt* planned_stmt, const int rt_index, AttrNumber attnum) -> std::string;
auto get_table_oid_from_rte(const PlannedStmt* current_planned_stmt, const int varno) -> Oid;
auto is_column_nullable(const PlannedStmt* planned_stmt, const int rt_index, const AttrNumber attnum) -> bool;

// Translation helper for constants
auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;

// Get all columns from a table
auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, const int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast