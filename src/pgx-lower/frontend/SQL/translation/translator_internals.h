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
#include <utility>

// ===========================================================================
// Translation Context
// ===========================================================================

namespace pgx_lower::frontend::sql {

// TODO: NV Improve state correctness
// Our data model is currently a mess. I believe we should change to
// QueryState
//      QueryContext -> represents global/absolute state/variables.
//      TranslationResult -> represents relative state/variables
//
// For the QueryContext, you can ADD information to it, for instance the subquery
// node might have to add information about the range table lookups because it created these
// global columns. However, you cannot update or modify entries inside of it after its
// created
//
// For TranslationResult, there are a number of challenges. We need it to be immutable, and
// construction guarantees correctness. This means we need to either introduce a factory pattern
// or add like... 5-10 constructors.
//
// There's essentially three types of TranslationResults. The first represents reading from a leaf node
// or absolute positions, subquery columns. These require a lot of validation that what they're being made
// from is valid.
// The second is joining two TranslationResults. So, if you have an expression inside of your Join node,
// this requires a TranslationResult from the children of the Join node. This has like -1 and -2 for left/right
// children. These are constructed from 2+ TranslationResults, and basically relies on induction.
// The third is carried up. For instance, if you have a Projection node with a child, it chops columns
// out of its child. These should be able to validate with the result list of the ProjectionNode.
//
// All of these can be validate with MLIR results, like the joins can validate by binding MLIR graph structures
// by passing in the left and right mlir::Values. So, maybe our constructors for a lot of these will look like
// construct([TranslationResult], [mlir::Value], result list, ... other things)
//
// With all of this, we know that our TranslationResult will always represent something VALID. However,
// we won't know whether it belongs to our current expression. This I'm not sure we need to worry about, or
// rather, introducing a solution adds too much complexity to justify the benefit compared to testing.
template<typename T>
using OptRefT = std::optional<std::reference_wrapper<T>>;

struct ColumnInfo {
    std::string name;
    unsigned int type_oid;
    int32_t typmod;
    bool nullable;

    ColumnInfo(std::string n, const unsigned int T, const int32_t TYPEMOD, const bool nullable)
    : name(std::move(n))
    , type_oid(T)
    , typmod(TYPEMOD)
    , nullable(nullable) {}
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
        [[nodiscard]] auto toString() const -> std::string {
            return "ColumnSchema(table='" + table_name + "', column='" + column_name
                   + "', oid=" + std::to_string(type_oid) + ", typmod=" + std::to_string(typmod)
                   + ", nullable=" + (nullable ? "true" : "false") + ")";
        }
    };

    std::vector<ColumnSchema> columns;
    std::string current_scope;
    size_t left_child_column_count = 0;

    [[nodiscard]] auto toString() const -> std::string {
        std::string result = "TranslationResult(op=" + (op ? std::to_string(reinterpret_cast<uintptr_t>(op)) : "null")
                             + ", scope=" + current_scope + ", columns=[";
        for (const auto& col : columns) {
            result += "\n\t" + col.toString();
        }
        result += "]";

        return result;
    }
};

/**
 * This class aims to contain IMMUTABLE state except for the OpBuilder. Since the OpBuilder is effectively our primary
 * goal in this crawl, It's ok to constantly mutate it. However, everything else here should be immutable.
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
    mlir::Value current_tuple;
    mlir::Value outer_tuple;
    OptRefT<const TranslationResult> outer_result;
    std::unordered_map<int, TranslationResult> init_plan_results;
    std::unordered_map<int, SubqueryInfo> subquery_param_mapping;
    std::map<std::pair<int, int>, std::pair<std::string, std::string>> varno_resolution;

    struct CorrelationInfo {
        std::string table_scope;
        std::string column_name;
        bool nullable;
    };
    std::unordered_map<int, CorrelationInfo> correlation_params;
    std::unordered_map<int, Var*> nest_params;
    static int outer_join_counter;

    static TranslationContext createChildContext(const TranslationContext& parent) {
        return TranslationContext{
            parent.current_stmt,     parent.builder,           parent.current_module,    parent.current_tuple,
            parent.current_tuple,    parent.outer_result,      parent.init_plan_results, parent.subquery_param_mapping,
            parent.varno_resolution, parent.correlation_params, parent.nest_params};
    }

    static TranslationContext createChildContext(const TranslationContext& parent, mlir::OpBuilder& new_builder,
                                                 mlir::Value new_current_tuple) {
        return TranslationContext{parent.current_stmt,        new_builder,              parent.current_module,
                                  new_current_tuple,          parent.current_tuple,     parent.outer_result,
                                  parent.init_plan_results,   parent.subquery_param_mapping,
                                  parent.varno_resolution,    parent.correlation_params, parent.nest_params};
    }

    [[nodiscard]] auto resolve_var(int varno, int varattno) const -> std::optional<std::pair<std::string, std::string>> {
        const auto KEY = std::make_pair(varno, varattno);
        const auto it = varno_resolution.find(KEY);
        if (it != varno_resolution.end()) {
            return it->second;
        }
        return std::nullopt;
    }
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

    mlir::Value translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr,
                                        OptRefT<const TranslationResult> current_result = std::nullopt);
    auto translate_expression(const QueryCtxT& ctx, Expr* expr,
                              OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_expression_with_join_context(const QueryCtxT& ctx, Expr* expr, const TranslationResult* left_child,
                                                const TranslationResult* right_child) -> mlir::Value;
    auto translate_expression_for_stream(const QueryCtxT& ctx, Expr* expr, const TranslationResult& child_result,
                                         const std::string& suggested_name)
        -> pgx_lower::frontend::sql::StreamExpressionResult;

    auto translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr,
                           OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_var(const QueryCtxT& ctx, const Var* var,
                       OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_const(const QueryCtxT& ctx, Const* const_node,
                         OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr,
                             OptRefT<const TranslationResult> current_result = std::nullopt,
                             std::optional<std::vector<mlir::Value>> pre_translated_args = std::nullopt) -> mlir::Value;

    auto translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr,
                             OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_null_test(const QueryCtxT& ctx, const NullTest* null_test,
                             OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_aggref(const QueryCtxT& ctx, const Aggref* aggref,
                          OptRefT<const TranslationResult> current_result = std::nullopt) const -> mlir::Value;
    auto translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr,
                                 OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_scalar_array_op_expr(const QueryCtxT& ctx, const ScalarArrayOpExpr* scalar_array_op,
                                        OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr,
                             OptRefT<const TranslationResult> current_result = std::nullopt) -> mlir::Value;
    auto translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr, mlir::Value case_test_value)
        -> mlir::Value;

    // Subquery translation
    auto translate_subplan(const QueryCtxT& ctx, const SubPlan* subplan, OptRefT<const TranslationResult> current_result)
        -> mlir::Value;
    auto translate_subquery_plan(const QueryCtxT& parent_ctx, Plan* subquery_plan, const PlannedStmt* parent_stmt)
        -> std::pair<mlir::Value, TranslationResult>;
    auto translate_param(const QueryCtxT& ctx, const Param* param, OptRefT<const TranslationResult> current_result) const
        -> mlir::Value;

    // Plan node translation methods
    auto translate_plan_node(QueryCtxT& ctx, Plan* plan) -> TranslationResult;
    auto translate_seq_scan(QueryCtxT& ctx, SeqScan* seqScan) -> TranslationResult;
    auto translate_index_scan(QueryCtxT& ctx, IndexScan* indexScan) -> TranslationResult;
    auto translate_agg(QueryCtxT& ctx, const Agg* agg) -> TranslationResult;
    auto translate_sort(QueryCtxT& ctx, const Sort* sort) -> TranslationResult;
    auto translate_limit(QueryCtxT& ctx, const Limit* limit) -> TranslationResult;
    auto translate_gather(QueryCtxT& ctx, const Gather* gather) -> TranslationResult;
    auto translate_merge_join(QueryCtxT& ctx, MergeJoin* mergeJoin) -> TranslationResult;
    auto translate_hash_join(QueryCtxT& ctx, HashJoin* hashJoin) -> TranslationResult;
    auto translate_hash(QueryCtxT& ctx, const Hash* hash) -> TranslationResult;
    auto translate_nest_loop(QueryCtxT& ctx, NestLoop* nestLoop) -> TranslationResult;
    auto translate_material(QueryCtxT& ctx, const Material* material) -> TranslationResult;
    auto translate_subquery_scan(QueryCtxT& ctx, SubqueryScan* subqueryScan) -> TranslationResult;
    auto translate_cte_scan(QueryCtxT& ctx, CteScan* cteScan) -> TranslationResult;

    // InitPlan helpers
    auto process_init_plans(QueryCtxT& ctx, const Plan* plan) -> void;

    // Query function generation
    static auto create_query_function(mlir::OpBuilder& builder) -> mlir::func::FuncOp;
    auto generate_rel_alg_operations(const PlannedStmt* planned_stmt, QueryCtxT& context) -> bool;

    // Relational operation helpers

    auto apply_selection_from_qual(const QueryCtxT& ctx, const TranslationResult& input, const List* qual)
        -> TranslationResult;

    auto apply_selection_from_qual_with_columns(const QueryCtxT& ctx, const TranslationResult& input, const List* qual,
                                                const TranslationResult* merged_join_child) -> TranslationResult;
    auto apply_projection_from_target_list(const QueryCtxT& ctx, const TranslationResult& input, const List* target_list,
                                           const TranslationResult* merged_join_child = nullptr) -> TranslationResult;
    auto apply_projection_from_translation_result(const QueryCtxT& ctx, const TranslationResult& input,
                                                  const TranslationResult& merged_join_child, const List* target_list)
        -> TranslationResult;

    auto create_materialize_op(const QueryCtxT& context, mlir::Value tuple_stream,
                               const TranslationResult& translation_result) const -> mlir::Value;

    auto create_join_operation(QueryCtxT& ctx, JoinType join_type, mlir::Value left_value,
                               mlir::Value right_value, const TranslationResult& left_translation,
                               const TranslationResult& right_translation, List* join_clauses) -> TranslationResult;

    static auto merge_translation_results(const TranslationResult* left_child, const TranslationResult* right_child)
        -> TranslationResult;

    // Operation translation helpers
    auto extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr,
                                  OptRefT<const TranslationResult> current_result = std::nullopt)
        -> std::optional<std::pair<mlir::Value, mlir::Value>>;
    static auto normalize_bpchar_operands(const QueryCtxT& ctx, const OpExpr* op_expr, mlir::Value lhs, mlir::Value rhs)
        -> std::pair<mlir::Value, mlir::Value>;
    static auto translate_arithmetic_op(const QueryCtxT& ctx, const OpExpr* op_expr, mlir::Value lhs, mlir::Value rhs)
        -> mlir::Value;
    static auto upcast_binary_operation(const QueryCtxT& ctx, mlir::Value lhs, mlir::Value rhs)
        -> std::pair<mlir::Value, mlir::Value>;
    static auto translate_comparison_op(const QueryCtxT& context, Oid op_oid, mlir::Value lhs, mlir::Value rhs)
        -> mlir::Value;

    static auto verify_and_print(mlir::Value val) -> void;
    static void print_type(mlir::Type val);

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

    [[nodiscard]] auto map_postgre_sqltype(Oid type_oid, int32_t typmod, bool nullable = false) const -> mlir::Type;

    // Type modifier extraction functions
    static auto extract_numeric_info(int32_t typmod) -> std::pair<int32_t, int32_t>;
    static auto extract_timestamp_precision(int32_t typmod) -> mlir::db::TimeUnitAttr;
    static auto extract_varchar_length(int32_t typmod) -> int32_t;

   private:
    mlir::MLIRContext& context_;
};

// ===========================================================================
// Helper Functions
// ===========================================================================

// Schema access helpers (standalone functions for backwards compatibility)
auto get_table_name_from_rte(const PlannedStmt* current_planned_stmt, int varno) -> std::string;
auto get_table_alias_from_rte(const PlannedStmt* current_planned_stmt, int varno) -> std::string;
auto get_column_name_from_schema(const PlannedStmt* planned_stmt, int rt_index, AttrNumber attnum) -> std::string;
auto get_table_oid_from_rte(const PlannedStmt* current_planned_stmt, int varno) -> Oid;
auto is_column_nullable(const PlannedStmt* planned_stmt, int rt_index, AttrNumber attnum) -> bool;

// Translation helper for constants
auto translate_const(Const* const_node, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;

// Get all columns from a table
auto get_all_table_columns_from_schema(const PlannedStmt* current_planned_stmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast