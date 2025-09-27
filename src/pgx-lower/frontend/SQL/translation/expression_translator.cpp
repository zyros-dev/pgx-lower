#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "catalog/pg_type.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/utility/util_functions.h"

#include <algorithm>
#include <vector>
#include <sstream>
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/Column.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

namespace mlir::relalg {
class GetColumnOp;
} // namespace mlir::relalg

namespace postgresql_ast {
using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLASTTranslator::Impl::translate_expression(const QueryCtxT& ctx, Expr* expr,
                                                         OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Parsing %d", expr->type);

    if (!expr) {
        PGX_ERROR("Expression is null");
        throw std::runtime_error("Expression cannot be null");
    }

    switch (expr->type) {
    case T_Var: return translate_var(ctx, reinterpret_cast<Var*>(expr), current_result);
    case T_Const: return translate_const(ctx, reinterpret_cast<Const*>(expr), current_result);
    case T_OpExpr: return translate_op_expr(ctx, reinterpret_cast<OpExpr*>(expr), current_result);
    case T_FuncExpr: return translate_func_expr(ctx, reinterpret_cast<FuncExpr*>(expr), current_result);
    case T_BoolExpr: return translate_bool_expr(ctx, reinterpret_cast<BoolExpr*>(expr), current_result);
    case T_Aggref: return translate_aggref(ctx, reinterpret_cast<Aggref*>(expr), current_result);
    case T_NullTest: return translate_null_test(ctx, reinterpret_cast<NullTest*>(expr), current_result);
    case T_CoalesceExpr: return translate_coalesce_expr(ctx, reinterpret_cast<CoalesceExpr*>(expr), current_result);
    case T_ScalarArrayOpExpr: return translate_scalar_array_op_expr(ctx, reinterpret_cast<ScalarArrayOpExpr*>(expr), current_result);
    case T_CaseExpr: return translate_case_expr(ctx, reinterpret_cast<CaseExpr*>(expr), current_result);
    case T_CoerceViaIO: return translate_coerce_via_io(ctx, expr, current_result);
    case T_RelabelType: {
        const auto* relabel = reinterpret_cast<RelabelType*>(expr);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unwrapping T_RelabelType to translate underlying expression");
        return translate_expression(ctx, relabel->arg, current_result);
    }
    case T_SubPlan:
        return translate_subplan(ctx, reinterpret_cast<SubPlan*>(expr), current_result);
    case T_Param:
        return translate_param(ctx, reinterpret_cast<Param*>(expr), current_result);
    default: {
        PGX_ERROR("Unsupported expression type: %d", expr->type);
        throw std::runtime_error("Unsupported expression type - read the logs");
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_expression_with_join_context(const QueryCtxT& ctx, Expr* expr,
                                                                           const TranslationResult* left_child,
                                                                           const TranslationResult* right_child,
                                                                           std::optional<mlir::Value> outer_tuple_arg)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!expr) {
        PGX_ERROR("Null expression");
        throw std::runtime_error("check logs");
    }

    if (expr->type == T_Var) {
        const auto* var = reinterpret_cast<const Var*>(expr);

        PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Processing Var: varno=%d, varattno=%d, varattnosyn=%d, vartype=%d",
                var->varno, var->varattno, var->varattnosyn, var->vartype);

        if (var->varno == OUTER_VAR && left_child) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] OUTER_VAR: left_child has %zu columns", left_child->columns.size());
            for (size_t i = 0; i < left_child->columns.size(); ++i) {
                const auto& c = left_child->columns[i];
                PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s (oid=%d)", i + 1, c.table_name.c_str(), c.column_name.c_str(), c.type_oid);
            }

            if (var->varattno > 0 && var->varattno <= static_cast<int>(left_child->columns.size())) {
                const auto& col = left_child->columns[var->varattno - 1];

                PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] OUTER_VAR varattno=%d resolved to %s.%s (position %d)",
                        var->varattno, col.table_name.c_str(), col.column_name.c_str(), var->varattno - 1);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Column details: type_oid=%d, nullable=%d",
                        col.type_oid, col.nullable);

                auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
                if (!dialect) {
                    PGX_ERROR("RelAlg dialect not registered");
                    throw std::runtime_error("Check logs");
                }

                auto& columnManager = dialect->getColumnManager();
                auto colRef = columnManager.createRef(col.table_name, col.column_name);
                colRef.getColumn().type = col.mlir_type;

                const auto tuple_to_use = outer_tuple_arg.value_or(ctx.current_tuple);
                auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(),
                                                                              col.mlir_type, colRef, tuple_to_use);

                return getColOp.getRes();
            } else {
                PGX_ERROR("OUTER_VAR varattno %d out of range (have %zu columns)", var->varattno,
                          left_child->columns.size());
                throw std::runtime_error("Check logs");
            }
        } else if (var->varno == INNER_VAR && right_child) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INNER_VAR: right_child has %zu columns", right_child->columns.size());
            for (size_t i = 0; i < right_child->columns.size(); ++i) {
                const auto& c = right_child->columns[i];
                PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s (oid=%d)", i + 1, c.table_name.c_str(), c.column_name.c_str(), c.type_oid);
            }

            const auto key = std::make_pair(var->varnosyn, var->varattno);
            const auto* resolved_col_ptr = [&]() -> const TranslationResult::ColumnSchema* {
                if (auto mapping = right_child->resolve_var(var->varnosyn, var->varattno)) {
                    const auto& [table_name, col_name] = *mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INNER_VAR using varno_resolution: varnosyn=%d, varattno=%d -> @%s::@%s",
                            var->varnosyn, var->varattno, table_name.c_str(), col_name.c_str());
                    for (const auto& c : right_child->columns) {
                        if (c.table_name == table_name && c.column_name == col_name) {
                            return &c;
                        }
                    }
                    PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Mapping found but column not in schema, falling back to position");
                }
                return nullptr;
            }();

            const auto& col = resolved_col_ptr ? *resolved_col_ptr
                            : (var->varattno > 0 && var->varattno <= static_cast<int>(right_child->columns.size()))
                              ? right_child->columns[var->varattno - 1]
                              : throw std::runtime_error("INNER_VAR varattno out of range");

            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INNER_VAR varattno=%d resolved to %s.%s",
                    var->varattno, col.table_name.c_str(), col.column_name.c_str());
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Column details: type_oid=%d, nullable=%d",
                    col.type_oid, col.nullable);

            auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
            if (!dialect) {
                PGX_ERROR("RelAlg dialect not registered");
                throw std::runtime_error("Check logs");
            }

            auto& columnManager = dialect->getColumnManager();
            auto colRef = columnManager.createRef(col.table_name, col.column_name);
            colRef.getColumn().type = col.mlir_type;

            auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(),
                                                                          col.mlir_type, colRef, ctx.current_tuple);

            return getColOp.getRes();
        } else {
            return translate_var(ctx, var);
        }
    }

    // For non-var, rely on the expression translator
    switch (expr->type) {
    case T_OpExpr: {
        const auto* op_expr = reinterpret_cast<const OpExpr*>(expr);
        return translate_op_expr_with_join_context(ctx, op_expr, left_child, right_child, outer_tuple_arg);
    }
    case T_BoolExpr: {
        const auto* bool_expr = reinterpret_cast<const BoolExpr*>(expr);
        return translate_bool_expr_with_join_context(ctx, bool_expr, left_child, right_child, outer_tuple_arg);
    }
    case T_FuncExpr: {
        const auto* func_expr = reinterpret_cast<const FuncExpr*>(expr);
        return translate_func_expr_with_join_context(ctx, func_expr, left_child, right_child, outer_tuple_arg);
    }
    default: return translate_expression(ctx, expr);
    }
}

auto PostgreSQLASTTranslator::Impl::translate_op_expr_with_join_context(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                                        const TranslationResult* left_child,
                                                                        const TranslationResult* right_child,
                                                                        std::optional<mlir::Value> outer_tuple_arg)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    PGX_LOG(AST_TRANSLATE, DEBUG, "[OPEXPR] Processing OpExpr with opno=%d, opfuncid=%d",
            op_expr->opno, op_expr->opfuncid);

    if (!op_expr->args || op_expr->args->length != 2) {
        PGX_ERROR("OpExpr should have exactly 2 arguments, got %d", op_expr->args ? op_expr->args->length : 0);
        throw std::runtime_error("Check logs");
    }

    auto* leftExpr = static_cast<Expr*>(lfirst(list_nth_cell(op_expr->args, 0)));
    auto* rightExpr = static_cast<Expr*>(lfirst(list_nth_cell(op_expr->args, 1)));

    PGX_LOG(AST_TRANSLATE, DEBUG, "[OPEXPR] Left operand type: %d, Right operand type: %d",
            leftExpr ? leftExpr->type : -1, rightExpr ? rightExpr->type : -1);

    const auto lhs = translate_expression_with_join_context(ctx, leftExpr, left_child, right_child, outer_tuple_arg);
    const auto rhs = translate_expression_with_join_context(ctx, rightExpr, left_child, right_child, outer_tuple_arg);

    if (!lhs || !rhs) {
        PGX_ERROR("Failed to translate OpExpr operands - lhs=%p, rhs=%p", lhs.getAsOpaquePointer(), rhs.getAsOpaquePointer());
        throw std::runtime_error("Check logs");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[OPEXPR] Successfully translated both operands");

    const auto op_oid = op_expr->opno;

    if (const auto result = translate_arithmetic_op(ctx, op_expr, lhs, rhs)) {
        return result;
    }

    if (const auto result = translate_comparison_op(ctx, op_oid, lhs, rhs)) {
        return result;
    }

    if (auto* oprname = get_opname(op_oid)) {
        std::string op(oprname);
        pfree(oprname);

        if (op == "~~") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LIKE operator to db.runtime_call");

            auto convertedLhs = lhs;
            auto convertedRhs = rhs;

            const auto lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
            const auto rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

            if (lhsNullable && !rhsNullable) {
                auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
                convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType,
                                                                          rhs);
            } else if (!lhsNullable && rhsNullable) {
                auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
                convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType,
                                                                          lhs);
            }

            const bool hasNullableOperand = lhsNullable || rhsNullable;
            auto resultType = hasNullableOperand ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                                                          ctx.builder.getI1Type()))
                                                 : mlir::Type(ctx.builder.getI1Type());

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                 ctx.builder.getStringAttr("Like"),
                                                                 mlir::ValueRange{convertedLhs, convertedRhs});
            return op2.getRes();
        } else if (op == "!~~") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NOT LIKE operator to negated db.runtime_call");

            auto convertedLhs = lhs;
            auto convertedRhs = rhs;

            const auto lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
            const auto rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

            if (lhsNullable && !rhsNullable) {
                auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
                convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType,
                                                                          rhs);
            } else if (!lhsNullable && rhsNullable) {
                auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
                convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType,
                                                                          lhs);
            }

            const bool hasNullableOperand = lhsNullable || rhsNullable;
            auto resultType = hasNullableOperand ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                                                          ctx.builder.getI1Type()))
                                                 : mlir::Type(ctx.builder.getI1Type());

            auto likeOp = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                    ctx.builder.getStringAttr("Like"),
                                                                    mlir::ValueRange{convertedLhs, convertedRhs});
            auto notOp = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), resultType, likeOp.getRes());
            return notOp.getRes();
        } else if (op == "||") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating || operator to StringRuntime::concat");

            const bool hasNullableOperand = isa<mlir::db::NullableType>(lhs.getType())
                                            || isa<mlir::db::NullableType>(rhs.getType());

            auto resultType = hasNullableOperand
                                  ? mlir::Type(mlir::db::NullableType::get(
                                        ctx.builder.getContext(), mlir::db::StringType::get(ctx.builder.getContext())))
                                  : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(
                ctx.builder.getUnknownLoc(), resultType, ctx.builder.getStringAttr("Concat"), mlir::ValueRange{lhs, rhs});

            return op2.getRes();
        }
    }

    PGX_ERROR("Unsupported operator OID: %u", op_oid);
    throw std::runtime_error("Check logs");
}

auto PostgreSQLASTTranslator::Impl::translate_bool_expr_with_join_context(const QueryCtxT& ctx, const BoolExpr* bool_expr,
                                                                          const TranslationResult* left_child,
                                                                          const TranslationResult* right_child,
                                                                          std::optional<mlir::Value> outer_tuple_arg)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!bool_expr->args || bool_expr->args->length == 0) {
        PGX_ERROR("BoolExpr has no arguments");
        throw std::runtime_error("Check logs");
    }

    std::vector<mlir::Value> operands;
    ListCell* lc;
    foreach (lc, bool_expr->args) {
        auto* arg = static_cast<Expr*>(lfirst(lc));
        if (auto operand = translate_expression_with_join_context(ctx, arg, left_child, right_child, outer_tuple_arg)) {
            if (!operand.getType().isInteger(1)) {
                operand = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), operand);
            }
            operands.push_back(operand);
        } else {
            PGX_ERROR("Failed to translate BoolExpr argument");
            throw std::runtime_error("Check logs");
        }
    }

    if (operands.empty()) {
        PGX_ERROR("No valid operands for BoolExpr");
        throw std::runtime_error("Check logs");
    }

    mlir::Value result = operands[0];
    for (size_t i = 1; i < operands.size(); ++i) {
        switch (bool_expr->boolop) {
        case AND_EXPR:
            result = ctx.builder.create<mlir::db::AndOp>(ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
                                                         mlir::ValueRange{result, operands[i]});
            break;
        case OR_EXPR:
            result = ctx.builder.create<mlir::db::OrOp>(ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
                                                        mlir::ValueRange{result, operands[i]});
            break;
        default: PGX_ERROR("Unsupported BoolExpr type: %d", bool_expr->boolop); throw std::runtime_error("Check logs");
        }
    }

    if (bool_expr->boolop == NOT_EXPR && operands.size() == 1) {
        result = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), operands[0]);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_func_expr_with_join_context(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                                                          const TranslationResult* left_child,
                                                                          const TranslationResult* right_child,
                                                                          std::optional<mlir::Value> outer_tuple_arg)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!func_expr) {
        PGX_ERROR("Invalid FuncExpr parameters");
        throw std::runtime_error("Invalid FuncExpr parameters");
    }

    auto args = std::vector<mlir::Value>{};
    if (func_expr->args && func_expr->args->length > 0) {
        if (!func_expr->args->elements) {
            PGX_ERROR("FuncExpr args list has length but no elements array");
            throw std::runtime_error("FuncExpr args list has length but no elements array");
        }

        ListCell* lc;
        foreach (lc, func_expr->args) {
            if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                if (mlir::Value argValue = translate_expression_with_join_context(ctx, reinterpret_cast<Expr*>(argNode),
                                                                                  left_child, right_child, outer_tuple_arg)) {
                    args.push_back(argValue);
                }
            }
        }
    }

    const auto loc = ctx.builder.getUnknownLoc();

    char* funcname = get_func_name(func_expr->funcid);
    if (!funcname) {
        PGX_ERROR("Unknown function OID %d", func_expr->funcid);
        throw std::runtime_error("Unknown function OID " + std::to_string(func_expr->funcid));
    }

    std::string func(funcname);
    pfree(funcname);

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating function %s with join context", func.c_str());

    if (func == "date_part" || func == "extract") {
        if (args.size() != 2) {
            PGX_ERROR("EXTRACT/date_part requires exactly 2 arguments, got %zu", args.size());
            throw std::runtime_error("EXTRACT/date_part requires exactly 2 arguments");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating EXTRACT function");
        auto runtimeCall = ctx.builder.create<mlir::db::RuntimeCall>(
            loc, ctx.builder.getI64Type(), "ExtractFromDate", args);
        return runtimeCall.getRes();
    }

    return translate_func_expr(ctx, func_expr, std::nullopt);
}

mlir::Value PostgreSQLASTTranslator::Impl::translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr, OptRefT<const TranslationResult> current_result) {
    // Handle type coercion via I/O functions (e.g., int::text)
    const auto* coerce = reinterpret_cast<CoerceViaIO*>(expr);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Processing T_CoerceViaIO to type OID %d", coerce->resulttype);

    auto argValue = translate_expression(ctx, coerce->arg, current_result);
    if (!argValue) {
        PGX_ERROR("Failed to translate CoerceViaIO argument");
        throw std::runtime_error("Failed to translate CoerceViaIO argument");
    }

    const bool isNullable = mlir::isa<mlir::db::NullableType>(argValue.getType());
    const auto type_mapper = PostgreSQLTypeMapper(context_);
    auto targetType = type_mapper.map_postgre_sqltype(coerce->resulttype, -1, isNullable);

    return ctx.builder.create<mlir::db::CastOp>(ctx.builder.getUnknownLoc(), targetType, argValue);
}

auto PostgreSQLASTTranslator::Impl::translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                      OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!op_expr) {
        PGX_ERROR("Invalid OpExpr parameters");
        throw std::runtime_error("Invalid OpExpr parameters");
    }

    auto operands = extract_op_expr_operands(ctx, op_expr, current_result);
    if (!operands) {
        PGX_ERROR("Failed to extract OpExpr operands");
        throw std::runtime_error("Invalid OpExpr parameters");
    }

    const auto [lhs, rhs] = *operands;
    const Oid opOid = op_expr->opno;

    {
        if (auto result = translate_arithmetic_op(ctx, op_expr, lhs, rhs))
            return result;
    }

    {
        if (auto result = translate_comparison_op(ctx, opOid, lhs, rhs))
            return result;
    }

    if (auto* oprname = get_opname(opOid)) {
        std::string op(oprname);
        pfree(oprname);

        if (op == "~~") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LIKE operator to db.runtime_call");

            auto convertedLhs = lhs;
            auto convertedRhs = rhs;

            const auto lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
            const auto rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

            if (lhsNullable && !rhsNullable) {
                auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
                convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType,
                                                                          rhs);
            } else if (!lhsNullable && rhsNullable) {
                auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
                convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType,
                                                                          lhs);
            }

            const bool hasNullableOperand = lhsNullable || rhsNullable;
            auto resultType = hasNullableOperand ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                                                          ctx.builder.getI1Type()))
                                                 : mlir::Type(ctx.builder.getI1Type());

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                 ctx.builder.getStringAttr("Like"),
                                                                 mlir::ValueRange{convertedLhs, convertedRhs});

            return op2.getRes();
        } else if (op == "!~~") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NOT LIKE operator to negated db.runtime_call");
            auto convertedLhs = lhs;
            auto convertedRhs = rhs;

            const bool lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
            const bool rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

            if (lhsNullable && !rhsNullable) {
                auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
                convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType,
                                                                          rhs);
            } else if (!lhsNullable && rhsNullable) {
                auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
                convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType,
                                                                          lhs);
            }

            const mlir::Type boolType = ctx.builder.getI1Type();
            auto resultType = (lhsNullable || rhsNullable)
                                  ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), boolType))
                                  : boolType;

            auto likeOp = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                    ctx.builder.getStringAttr("Like"),
                                                                    mlir::ValueRange{convertedLhs, convertedRhs});

            auto notOp = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), resultType, likeOp.getRes());

            return notOp.getResult();
        } else if (op == "||") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating || operator to StringRuntime::concat");

            const bool hasNullableOperand = isa<mlir::db::NullableType>(lhs.getType())
                                            || isa<mlir::db::NullableType>(rhs.getType());

            auto resultType = hasNullableOperand
                                  ? mlir::Type(mlir::db::NullableType::get(
                                        ctx.builder.getContext(), mlir::db::StringType::get(ctx.builder.getContext())))
                                  : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(
                ctx.builder.getUnknownLoc(), resultType, ctx.builder.getStringAttr("Concat"), mlir::ValueRange{lhs, rhs});

            return op2.getRes();
        }
    }

    PGX_ERROR("Unsupported operator OID: %d", opOid);
    throw std::runtime_error("Unsupported operator");
}

auto PostgreSQLASTTranslator::Impl::translate_var(const QueryCtxT& ctx, const Var* var,
                                                  OptRefT<const TranslationResult> current_result) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!var || !ctx.current_tuple) {
        PGX_ERROR("Invalid Var parameters: var=%p, builder=%p, tuple=%p\n%s", var, ctx.builder,
                  ctx.current_tuple.getAsOpaquePointer());
        throw std::runtime_error("Invalid Var parameters");
    }

    if (ctx.current_tuple) {
        int actualVarno = var->varno;

        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_var: varno=%d, varattno=%d", var->varno, var->varattno);

        std::string tableName, colName;
        bool nullable;

        if (current_result && current_result->get().resolve_var(actualVarno, var->varattno)) {
            const auto& [mappedTable, mappedColumn] = *current_result->get().resolve_var(actualVarno, var->varattno);
            tableName = mappedTable;
            colName = mappedColumn;
            nullable = is_column_nullable(&ctx.current_stmt, actualVarno, var->varattno);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Using TranslationResult mapping for varno=%d, varattno=%d -> (%s, %s)",
                    actualVarno, var->varattno, tableName.c_str(), colName.c_str());
        } else if (var->varno == OUTER_VAR) {
            if (!current_result || var->varattno <= 0 ||
                var->varattno > static_cast<int>(current_result->get().columns.size())) {
                PGX_ERROR("OUTER_VAR varattno=%d out of range (child has %zu columns)",
                          var->varattno, current_result ? current_result->get().columns.size() : 0);
                throw std::runtime_error("OUTER_VAR reference without valid child result");
            }
            const auto& col = current_result->get().columns[var->varattno - 1];
            tableName = col.table_name;
            colName = col.column_name;
            nullable = col.nullable;
            PGX_LOG(AST_TRANSLATE, DEBUG, "OUTER_VAR varattno=%d resolved to %s.%s (nullable=%d) from child result",
                    var->varattno, tableName.c_str(), colName.c_str(), nullable);
        } else {
            tableName = get_table_alias_from_rte(&ctx.current_stmt, actualVarno);
            colName = get_column_name_from_schema(&ctx.current_stmt, actualVarno, var->varattno);
            nullable = is_column_nullable(&ctx.current_stmt, actualVarno, var->varattno);
        }

        auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        if (!dialect) {
            PGX_ERROR("RelAlg dialect not registered");
            throw std::runtime_error("Check logs");
        }

        auto& columnManager = dialect->getColumnManager();

        const auto type_mapper = PostgreSQLTypeMapper(context_);
        auto mlirType = type_mapper.map_postgre_sqltype(var->vartype, var->vartypmod, nullable);

        auto colRef = columnManager.createRef(tableName, colName);
        colRef.getColumn().type = mlirType;

        auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), mlirType, colRef,
                                                                      ctx.current_tuple);

        return getColOp.getRes();
    } else {
        PGX_ERROR("No tuple context for Var translation");
        throw std::runtime_error("No tuple context for Var translation");
    }
}

auto PostgreSQLASTTranslator::Impl::translate_const(const QueryCtxT& ctx, Const* const_node,
                                                    OptRefT<const TranslationResult> current_result) const
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    (void)current_result;
    return postgresql_ast::translate_const(const_node, ctx.builder, context_);
}

auto PostgreSQLASTTranslator::Impl::translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                                        OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!func_expr) {
        PGX_ERROR("Invalid FuncExpr parameters");
        throw std::runtime_error("Invalid FuncExpr parameters");
    }

    auto args = std::vector<mlir::Value>{};
    if (func_expr->args && func_expr->args->length > 0) {
        if (!func_expr->args->elements) {
            PGX_ERROR("FuncExpr args list has length but no elements array");
            throw std::runtime_error("FuncExpr args list has length but no elements array");
        }

        ListCell* lc;
        foreach (lc, func_expr->args) {
            if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result)) {
                    args.push_back(argValue);
                }
            }
        }
    }

    const auto loc = ctx.builder.getUnknownLoc();

    char* funcname = get_func_name(func_expr->funcid);
    if (!funcname) {
        PGX_ERROR("Unknown function OID %d", func_expr->funcid);
        throw std::runtime_error("Unknown function OID " + std::to_string(func_expr->funcid));
    }

    std::string func(funcname);
    pfree(funcname);

    // // TODO - This is implemented in lingodb
    // if (func == "pg_catalog") {
    //     func = reinterpret_cast<value*>(funcCall->funcname_->tail->data.ptr_value)->val_.str_;
    // }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating function %s", func.c_str());
    if (func == "abs") {
        if (args.size() != 1) {
            PGX_ERROR("ABS requires exactly 1 argument, got %d", args.size());
            throw std::runtime_error("ABS requires exactly 1 argument");
        }

        std::string valueStr;
        llvm::raw_string_ostream stream(valueStr);
        args[0].print(stream);
        PGX_LOG(AST_TRANSLATE, DEBUG, "ABS argument: %s", valueStr.c_str());

        std::string typeStr;
        llvm::raw_string_ostream typeStream(typeStr);
        args[0].getType().print(typeStream);
        PGX_LOG(AST_TRANSLATE, DEBUG, "ABS argument type: %s", typeStr.c_str());

        // Choose the right abs function based on type
        auto baseType = getBaseType(args[0].getType());
        auto absFunctionName = "AbsInt";
        if (mlir::isa<mlir::db::DecimalType>(baseType)) {
            absFunctionName = "AbsDecimal";
            PGX_LOG(AST_TRANSLATE, DEBUG, "Using AbsDecimal for decimal type");
        }

        auto runtimeCall = ctx.builder.create<mlir::db::RuntimeCall>(loc, args[0].getType(), absFunctionName, args[0]);
        return runtimeCall.getRes();
    } else if (func == "upper") {
        if (args.size() != 1) {
            PGX_ERROR("UPPER requires exactly 1 argument");
            throw std::runtime_error("Check logs");
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating UPPER function to StringRuntime::upper");

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        mlir::Type resultType = hasNullableOperand
                                    ? mlir::Type(mlir::db::NullableType::get(
                                          ctx.builder.getContext(), mlir::db::StringType::get(ctx.builder.getContext())))
                                    : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc, resultType, ctx.builder.getStringAttr("Upper"),
                                                            mlir::ValueRange{args[0]});
        return op.getRes();
    } else if (func == "lower") {
        if (args.size() != 1) {
            PGX_ERROR("LOWER requires exactly 1 argument");
            throw std::runtime_error("Check logs");
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LOWER function to StringRuntime::lower");

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        mlir::Type resultType = hasNullableOperand
                                    ? mlir::Type(mlir::db::NullableType::get(
                                          ctx.builder.getContext(), mlir::db::StringType::get(ctx.builder.getContext())))
                                    : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc, resultType, ctx.builder.getStringAttr("Lower"),
                                                            mlir::ValueRange{args[0]});
        return op.getRes();
    } else if (func == "substring" || func == "substr") {
        if (args.size() < 2 || args.size() > 3) {
            PGX_ERROR("SUBSTRING requires 2 or 3 arguments, got %d", args.size());
            throw std::runtime_error("Check logs");
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating SUBSTRING function to StringRuntime::substring");

        // SUBSTRING(string, start [, length]
        auto substringArgs = std::vector{args[0], args[1]};
        if (args.size() == 3) {
            substringArgs.push_back(args[2]);
        } else {
            auto maxLength = ctx.builder.create<mlir::arith::ConstantIntOp>(loc, std::numeric_limits<int32_t>::max(),
                                                                            ctx.builder.getI32Type());
            substringArgs.push_back(maxLength);
        }

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        const auto resultType = hasNullableOperand
                                    ? mlir::Type(mlir::db::NullableType::get(
                                          ctx.builder.getContext(), mlir::db::StringType::get(ctx.builder.getContext())))
                                    : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc, resultType, ctx.builder.getStringAttr("Substring"),
                                                            mlir::ValueRange{substringArgs});
        return op.getRes();
    } else if (func == "date_part") {
        // date_part('field', timestamp) extracts a field from a date/timestamp
        // e.g., date_part('year', TIMESTAMP '2024-01-15') returns 2024
        if (args.size() != 2) {
            PGX_ERROR("DATE_PART requires exactly 2 arguments, got %zu", args.size());
            throw std::runtime_error("Check logs");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating DATE_PART function to ExtractFromDate runtime call");

        // args[0] is the field to extract (e.g., 'year', 'month', 'day')
        // args[1] is the date/timestamp value
        mlir::Type resultType = ctx.builder.getI64Type();
        bool hasNullableOperand = false;
        for (const auto& arg : args) {
            if (isa<mlir::db::NullableType>(arg.getType())) {
                hasNullableOperand = true;
                break;
            }
        }
        if (hasNullableOperand) {
            resultType = mlir::db::NullableType::get(ctx.builder.getContext(), resultType);
        }
        auto op = ctx.builder.create<mlir::db::RuntimeCall>(
            loc, resultType, ctx.builder.getStringAttr("ExtractFromDate"), mlir::ValueRange{args[0], args[1]});

        return op.getRes();
    } else if (func == "numeric") {
        // numeric() function is used for casting to numeric/decimal type
        if (args.size() < 1 || args.size() > 3) {
            PGX_ERROR("NUMERIC cast requires 1-3 arguments, got %zu", args.size());
            throw std::runtime_error("Check logs");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NUMERIC cast");

        int precision = 38;
        int scale = 0;

        // PostgreSQL passes typmod as second argument which encodes (precision, scale)
        // The typmod encoding is: ((precision - 1) << 16) | (scale + VARHDRSZ)
        // where VARHDRSZ = 4
        if (args.size() >= 2) {
            if (auto* defOp = args[1].getDefiningOp()) {
                if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp)) {
                    int32_t typmod = constOp.value();

                    // Decode PostgreSQL numeric typmod
                    if (typmod >= 0) {
                        scale = (typmod & 0xFFFF) - 4;
                        precision = (typmod >> 16) & 0xFFFF;
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Decoded NUMERIC typmod %d to precision=%d, scale=%d", typmod,
                                precision, scale);
                    }
                }
            }
        }

        auto decimalType = mlir::db::DecimalType::get(ctx.builder.getContext(), precision, scale);
        const bool isNullable = mlir::isa<mlir::db::NullableType>(args[0].getType());
        auto targetType = isNullable ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), decimalType))
                                     : mlir::Type(decimalType);
        return ctx.builder.create<mlir::db::CastOp>(loc, targetType, args[0]);
    } else if (func == "varchar" || func == "text" || func == "char" || func == "bpchar") {
        if (args.empty()) {
            PGX_ERROR("%s requires at least 1 argument", func.c_str());
            throw std::runtime_error("Check logs");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating %s type conversion", func.c_str());

        auto inputType = args[0].getType();
        auto baseInputType = getBaseType(inputType);

        if (mlir::isa<mlir::db::StringType>(baseInputType)) {
            return args[0];
        } else {
            const bool isNullable = mlir::isa<mlir::db::NullableType>(inputType);
            auto stringType = mlir::db::StringType::get(ctx.builder.getContext());
            auto resultType = isNullable ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), stringType))
                                         : mlir::Type(stringType);

            auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc, resultType, ctx.builder.getStringAttr("ToString"),
                                                                mlir::ValueRange{args[0]});
            return op.getRes();
        }
    } else if (func == "int4" || func == "int8" || func == "float4" || func == "float8") {
        if (args.empty()) {
            PGX_ERROR("%s requires at least 1 argument", func.c_str());
            throw std::runtime_error("Check logs");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating %s type conversion", func.c_str());

        mlir::Type targetBaseType;
        if (func == "int4") {
            targetBaseType = ctx.builder.getI32Type();
        } else if (func == "int8") {
            targetBaseType = ctx.builder.getI64Type();
        } else if (func == "float4") {
            targetBaseType = ctx.builder.getF32Type();
        } else if (func == "float8") {
            targetBaseType = ctx.builder.getF64Type();
        } else {
            PGX_ERROR("Unknown numeric conversion function: %s", func.c_str());
            throw std::runtime_error("Unknown numeric conversion function");
        }

        const bool isNullable = mlir::isa<mlir::db::NullableType>(args[0].getType());
        auto targetType = isNullable ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), targetBaseType))
                                     : targetBaseType;

        return ctx.builder.create<mlir::db::CastOp>(loc, targetType, args[0]);
    } else {
        PGX_ERROR("Unsupported function '%s' (OID %d)", func.c_str(), func_expr->funcid);
        throw std::runtime_error("Unsupported function: " + func);
    }
}

auto PostgreSQLASTTranslator::Impl::translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr,
                                                        OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!bool_expr) {
        PGX_ERROR("Invalid BoolExpr parameters");
        throw std::runtime_error("Invalid BoolExpr parameters");
    }

    if (!bool_expr->args || bool_expr->args->length == 0) {
        PGX_ERROR("BoolExpr has no arguments");
        throw std::runtime_error("Invalid BoolExpr parameters");
    }

    switch (bool_expr->boolop) {
    case BOOL_AND_EXPR: {
        mlir::Value result = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            if (!bool_expr->args->elements) {
                PGX_ERROR("BoolExpr AND args list has length but no elements array");
                throw std::runtime_error("Invalid BoolExpr parameters");
            }

            ListCell* lc;
            foreach (lc, bool_expr->args) {
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result))
                    {
                        if (!argValue.getType().isInteger(1)) {
                            argValue = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        } else {
                            result = ctx.builder.create<mlir::db::AndOp>(
                                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), mlir::ValueRange{result, argValue});
                        }
                    }
                }
            }
        }

        if (!result) {
            PGX_ERROR("Failed to match an operator");
            throw std::runtime_error("Failed to match an operator");
        }
        return result;
    }

    case BOOL_OR_EXPR: {
        mlir::Value result = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            if (!bool_expr->args->elements) {
                PGX_ERROR("BoolExpr OR args list has length but no elements array");
                throw std::runtime_error("BoolExpr OR args list has length but no elements array");
            }

            ListCell* lc;
            foreach (lc, bool_expr->args) {
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    if (auto argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result)) {
                        if (!argValue.getType().isInteger(1)) { // Ensure boolean type
                            argValue = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        } else {
                            result = ctx.builder.create<mlir::db::OrOp>(
                                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), mlir::ValueRange{result, argValue});
                        }
                    }
                }
            }
        }

        if (!result) {
            PGX_ERROR("Failed BoolExpr");
            throw std::runtime_error("Failed BoolExpr");
        }
        return result;
    }

    case BOOL_NOT_EXPR: {
        mlir::Value argVal = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            if (const ListCell* lc = list_head(bool_expr->args)) {
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    argVal = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result);
                }
            }
        }

        if (!argVal) {
            PGX_ERROR("NOT expression has no valid argument, using placeholder");
            throw std::runtime_error("NOT expression has no valid argument, using placeholder");
        }

        if (!argVal.getType().isInteger(1)) {
            argVal = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argVal);
        }

        return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), argVal);
    }

    default: {
        PGX_ERROR("Unknown BoolExpr type: %d", bool_expr->boolop);
        throw std::runtime_error("Unknown BoolExpr type");
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_null_test(const QueryCtxT& ctx, const NullTest* null_test,
                                                                       OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!null_test) {
        PGX_ERROR("Invalid NullTest parameters");
        throw std::runtime_error("Invalid NullTest parameters");
    }

    auto* argNode = reinterpret_cast<Node*>(null_test->arg);
    auto argVal = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result);
    if (!argVal) {
        PGX_ERROR("Failed to translate NullTest argument");
        throw std::runtime_error("Failed to translate NullTest argument");
    }

    if (isa<mlir::db::NullableType>(argVal.getType())) {
        auto isNull = ctx.builder.create<mlir::db::IsNullOp>(ctx.builder.getUnknownLoc(), argVal);
        if (null_test->nulltesttype == PG_IS_NOT_NULL)
            return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), isNull);
        else
            return isNull;
    } else {
        return ctx.builder.create<mlir::db::ConstantOp>(
            ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
            ctx.builder.getIntegerAttr(ctx.builder.getI1Type(), null_test->nulltesttype == PG_IS_NOT_NULL));
    }
}

auto PostgreSQLASTTranslator::Impl::translate_aggref(const QueryCtxT& ctx, const Aggref* aggref,
                                                     OptRefT<const TranslationResult> current_result) const
    -> mlir::Value {
    // TODO: Clean...
    PGX_IO(AST_TRANSLATE);
    if (!aggref) {
        PGX_ERROR("Invalid Aggref parameters");
        throw std::runtime_error("Invalid Aggref parameters");
    }

    char* rawFuncName = get_func_name(aggref->aggfnoid);
    if (rawFuncName == nullptr) {
        PGX_ERROR("Unknown aggregate function OID: %u", aggref->aggfnoid);
        throw std::runtime_error("Unknown aggregate function OID");
    }
    const std::string funcName(rawFuncName);
    pfree(rawFuncName);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_aggref: Looking for Aggref with function %s (OID %u, aggno=%d, aggtype=%d)",
            funcName.c_str(), aggref->aggfnoid, aggref->aggno, aggref->aggtype);

    if (current_result) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Available varno_resolution mappings in translate_aggref:");
        for (const auto& [key, value] : current_result->get().varno_resolution) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "  varno=%d, attno=%d -> (%s, %s)",
                    key.first, key.second, value.first.c_str(), value.second.c_str());
        }
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "No TranslationResult available in translate_aggref");
    }

    std::string scopeName, columnName;

    bool found = false;
    PGX_LOG(AST_TRANSLATE, DEBUG, "Current result is [%s]", current_result ? current_result->get().toString().data() : "Nothing!");
    if (current_result) {
        auto resolved = current_result->get().resolve_var(-2, aggref->aggno);
        if (resolved) {
            scopeName = resolved->first;
            columnName = resolved->second;
            found = true;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Using TranslationResult mapping for aggregate aggno=%d -> (%s, %s)",
                    aggref->aggno, scopeName.c_str(), columnName.c_str());
        }
    }

    if (!found) {
        PGX_ERROR("No mapping found for aggregate aggno=%d", aggref->aggno);
        throw std::runtime_error("Aggregate reference not found in column mappings");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Found Aggref mapping: aggno=%d -> scope=%s, column=%s", aggref->aggno,
            scopeName.c_str(), columnName.c_str());

    auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        PGX_ERROR("RelAlg dialect not registered");
        throw std::runtime_error("RelAlg dialect not registered");
    }
    auto& columnManager = dialect->getColumnManager();

    // Create column reference using the constructed scope and column name
    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating Aggref to GetColumnOp: scope=%s, column=%s", scopeName.c_str(),
            columnName.c_str());
    auto colRef = columnManager.createRef(scopeName, columnName);

    // Get the actual type from the column that was created during aggregation
    auto resultType = colRef.getColumn().type;
    if (!resultType) {
        std::string errorMsg = "Aggregate column type not found in column manager for scope='" +
                              scopeName + "', column='" + columnName + "'";
        PGX_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), resultType, colRef,
                                                                  ctx.current_tuple);
    return getColOp.getRes();
}

auto PostgreSQLASTTranslator::Impl::translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr, OptRefT<const TranslationResult> current_result)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!coalesce_expr) {
        PGX_ERROR("Invalid CoalesceExpr parameters");
        throw std::runtime_error("Invalid CoalesceExpr parameters");
    }

    if (!coalesce_expr->args || coalesce_expr->args->length == 0) {
        // No arguments - return NULL with default type
        auto nullType = mlir::db::NullableType::get(&context_, ctx.builder.getI32Type());
        return ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), nullType);
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE has %d arguments", coalesce_expr->args->length);

    auto translatedArgs = std::vector<mlir::Value>{};

    ListCell* cell;
    foreach (cell, coalesce_expr->args) {
        const auto expr = static_cast<Expr*>(lfirst(cell));
        if (mlir::Value val = translate_expression(ctx, expr, current_result)) {
            translatedArgs.push_back(val);
        } else {
            PGX_ERROR("Failed to translate COALESCE argument");
            throw std::runtime_error("Failed to translate COALESCE argument");
        }
    }

    if (translatedArgs.empty()) {
        PGX_ERROR("All COALESCE arguments failed to translate");
        throw std::runtime_error("All COALESCE arguments failed to translate");
    }

    // Determine common type - Only create nullable result if at least one argument is nullable
    mlir::Type baseType = nullptr;
    for (const auto& arg : translatedArgs) {
        const auto argType = arg.getType();
        if (auto nullableType = dyn_cast<mlir::db::NullableType>(argType)) {
            if (!baseType) {
                baseType = nullableType.getType();
            }
        } else if (!baseType) {
            baseType = argType;
        }
    }

    // COALESCE should always produce nullable type in query contexts
    // Even when all inputs are non-nullable, the result needs nullable wrapper
    auto commonType = mlir::db::NullableType::get(&context_, baseType);
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE common type determined - forcing nullable for query context");
    for (auto& val : translatedArgs) {
        if (val.getType() != commonType) {
            // Need to convert to common type
            if (!isa<mlir::db::NullableType>(val.getType())) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "Wrapping non-nullable argument to match common nullable type");
                // Wrap non-nullable value in nullable type with explicit false null flag
                auto falseFlag = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(), 0, 1);
                val = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), commonType, val, falseFlag);
            }
        }
    }

    std::function<mlir::Value(size_t)> buildCoalesceRecursive = [&](const size_t index) -> mlir::Value {
        const auto loc = ctx.builder.getUnknownLoc();
        if (index >= translatedArgs.size() - 1) {
            return translatedArgs.back();
        }

        auto value = translatedArgs[index];
        auto isNull = ctx.builder.create<mlir::db::IsNullOp>(loc, value);
        auto isNotNull = ctx.builder.create<mlir::db::NotOp>(loc, isNull);

        // Create scf.IfOp with automatic region creation (safer than manual blocks)
        auto ifOp = ctx.builder.create<mlir::scf::IfOp>(loc, commonType, isNotNull, true);

        // Then block: yield current value
        auto& thenRegion = ifOp.getThenRegion();
        auto* thenBlock = &thenRegion.front();
        ctx.builder.setInsertionPointToEnd(thenBlock);

        // Cast value if needed
        mlir::Value thenValue = value;
        if (value.getType() != commonType && !isa<mlir::db::NullableType>(value.getType())) {
            auto falseFlag = ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
            thenValue = ctx.builder.create<mlir::db::AsNullableOp>(loc, commonType, value, falseFlag);
        }
        ctx.builder.create<mlir::scf::YieldOp>(loc, thenValue);

        // Else block: recursive call for remaining arguments
        auto& elseRegion = ifOp.getElseRegion();
        auto* elseBlock = &elseRegion.front();
        ctx.builder.setInsertionPointToEnd(elseBlock);
        auto elseValue = buildCoalesceRecursive(index + 1);
        ctx.builder.create<mlir::scf::YieldOp>(loc, elseValue);

        // Reset insertion point after the ifOp
        ctx.builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    };

    const auto result = buildCoalesceRecursive(0);

    const bool resultIsNullable = mlir::isa<mlir::db::NullableType>(result.getType());
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE final result is nullable: %d", resultIsNullable);

    // COALESCE always returns nullable type for query context compatibility
    const auto resultIsNullableType = isa<mlir::db::NullableType>(result.getType());
    PGX_LOG(AST_TRANSLATE, IO, "translate_coalesce_expr OUT: MLIR Value (nullable=%d)", resultIsNullableType);

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_scalar_array_op_expr(const QueryCtxT& ctx,
                                                                   const ScalarArrayOpExpr* scalar_array_op,
                                                                   OptRefT<const TranslationResult> current_result)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!scalar_array_op) {
        PGX_ERROR("Invalid ScalarArrayOpExpr parameters");
        throw std::runtime_error("Invalid ScalarArrayOpExpr parameters");
    }

    const auto* args = scalar_array_op->args;
    if (!args || args->length != 2) {
        PGX_ERROR("ScalarArrayOpExpr: Expected 2 arguments, got %d", args ? args->length : 0);
        throw std::runtime_error("ScalarArrayOpExpr: Expected 2 arguments");
    }

    const auto leftNode = static_cast<Node*>(lfirst(&args->elements[0]));
    auto leftValue = translate_expression(ctx, reinterpret_cast<Expr*>(leftNode), std::nullopt);
    if (!leftValue) {
        PGX_ERROR("Failed to translate left operand of IN expression");
        throw std::runtime_error("Failed to translate left operand of IN expression");
    }

    const auto rightNode = static_cast<Node*>(lfirst(&args->elements[1]));

    PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr: Right operand nodeTag = %d", nodeTag(rightNode));
    auto arrayElements = std::vector<mlir::Value>{};

    if (nodeTag(rightNode) == T_ArrayExpr) {
        const auto arrayExpr = reinterpret_cast<ArrayExpr*>(rightNode);
        if (const auto* elements = arrayExpr->elements) {
            ListCell* lc;
            foreach (lc, elements) {
                const auto elemNode = static_cast<Node*>(lfirst(lc));
                if (mlir::Value elemValue = translate_expression(ctx, reinterpret_cast<Expr*>(elemNode), std::nullopt)) {
                    arrayElements.push_back(elemValue);
                }
            }
        }
    } else if (nodeTag(rightNode) == T_Const) {
        if (const auto constNode = reinterpret_cast<Const*>(rightNode); constNode->consttype == INT4ARRAYOID) {
            const auto array = DatumGetArrayTypeP(constNode->constvalue);
            int nitems;
            Datum* values;
            bool* nulls;

            deconstruct_array(array, INT4OID, sizeof(int32), true, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    int32 intValue = DatumGetInt32(values[i]);
                    auto elemValue = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                                                    intValue, ctx.builder.getI32Type());
                    arrayElements.push_back(elemValue);
                }
            }
        } else if (constNode->consttype == PG_TEXT_ARRAY_OID) {
            const auto array = DatumGetArrayTypeP(constNode->constvalue);
            int nitems;
            Datum* values;
            bool* nulls;

            deconstruct_array(array, TEXTOID, -1, false, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    const auto textValue = DatumGetTextP(values[i]);
                    std::string str_value(VARDATA(textValue), VARSIZE(textValue) - VARHDRSZ);

                    auto elemValue = ctx.builder.create<mlir::db::ConstantOp>(
                        ctx.builder.getUnknownLoc(), ctx.builder.getType<mlir::db::StringType>(),
                        ctx.builder.getStringAttr(str_value));
                    arrayElements.push_back(elemValue);
                }
            }
        } else {
            PGX_WARNING("ScalarArrayOpExpr: Unsupported const array type %u", constNode->consttype);
        }
    } else if (nodeTag(rightNode) == T_SubPlan) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr with SubPlan operand detected (ANY/ALL/IN subquery)");
        throw std::runtime_error("UNEXPECTED: Is this possible?");
    } else if (nodeTag(rightNode) == T_Param) {
        const auto* param = reinterpret_cast<Param*>(rightNode);

        if (param->paramkind != PARAM_EXEC) {
            PGX_ERROR("Only PARAM_EXEC parameters are supported (got paramkind=%d)", param->paramkind);
            throw std::runtime_error("Unsupported param kind");
        }

        const auto& init_plan_results = ctx.init_plan_results;
        const auto it = init_plan_results.find(param->paramid);

        if (it == init_plan_results.end()) {
            PGX_ERROR("Param references unknown paramid=%d (InitPlan not processed?)", param->paramid);
            throw std::runtime_error("Param references unknown InitPlan result");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving ScalarArrayOpExpr Param paramid=%d to InitPlan result", param->paramid);

        const auto& initplan_result = it->second;

        if (!initplan_result.op) {
            PGX_ERROR("InitPlan result for paramid=%d has no operation", param->paramid);
            throw std::runtime_error("Invalid InitPlan result");
        }

        if (initplan_result.columns.empty()) {
            PGX_ERROR("InitPlan result for paramid=%d has no columns", param->paramid);
            throw std::runtime_error("InitPlan must return at least one column");
        }

        mlir::Value initplan_stream = initplan_result.op->getResult(0);
        const auto& initplan_column = initplan_result.columns[0];

        auto& col_mgr = ctx.builder.getContext()
            ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
            ->getColumnManager();

        const auto tuple_type = mlir::relalg::TupleType::get(ctx.builder.getContext());

        auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(
            ctx.builder.getUnknownLoc(), initplan_stream);

        auto& pred_region = selection_op.getPredicate();
        auto& pred_block = pred_region.emplaceBlock();
        auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

        mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

        auto initplan_col_ref = col_mgr.createRef(initplan_column.table_name, initplan_column.column_name);
        auto initplan_value = pred_builder.create<mlir::relalg::GetColumnOp>(
            pred_builder.getUnknownLoc(), initplan_column.mlir_type,
            initplan_col_ref, inner_tuple);

        char* oprname = get_opname(scalar_array_op->opno);
        if (!oprname) {
            PGX_ERROR("Unknown operator OID %u in ScalarArrayOpExpr with Param", scalar_array_op->opno);
            throw std::runtime_error("Unknown operator OID");
        }

        const std::string op(oprname);
        pfree(oprname);

        mlir::db::DBCmpPredicate predicate;
        if (op == "=") {
            predicate = mlir::db::DBCmpPredicate::eq;
        } else if (op == "<>" || op == "!=") {
            predicate = mlir::db::DBCmpPredicate::neq;
        } else if (op == "<") {
            predicate = mlir::db::DBCmpPredicate::lt;
        } else if (op == "<=") {
            predicate = mlir::db::DBCmpPredicate::lte;
        } else if (op == ">") {
            predicate = mlir::db::DBCmpPredicate::gt;
        } else if (op == ">=") {
            predicate = mlir::db::DBCmpPredicate::gte;
        } else {
            PGX_ERROR("Unsupported operator '%s' in ScalarArrayOpExpr with Param", op.c_str());
            throw std::runtime_error("Unsupported operator");
        }

        auto comparison = pred_builder.create<mlir::db::CmpOp>(
            pred_builder.getUnknownLoc(), predicate, leftValue, initplan_value);

        pred_builder.create<mlir::relalg::ReturnOp>(
            pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

        auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(
            ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
            selection_op.getResult());

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created EXISTS pattern for ScalarArrayOpExpr with Param");
        return exists_op.getResult();
    } else {
        PGX_ERROR("ScalarArrayOpExpr: Unexpected right operand type %d", nodeTag(rightNode));
        throw std::runtime_error("Unsupported ScalarArrayOpExpr operand type");
    }

    if (arrayElements.empty()) {
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                              scalar_array_op->useOr ? 0 : 1, ctx.builder.getI1Type());
    }

    mlir::Value result = nullptr;
    for (const auto& elemValue : arrayElements) {
        mlir::Value cmp = nullptr;

        char* oprname = get_opname(scalar_array_op->opno);
        if (!oprname) {
            PGX_WARNING("Unknown operator OID %u in IN expression, defaulting to equality", scalar_array_op->opno);
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      leftValue, elemValue);
        } else {
            std::string op(oprname);
            pfree(oprname);

            if (op == "=") {
                cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                          leftValue, elemValue);
            } else if (op == "<>" || op == "!=") {
                cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::neq,
                                                          leftValue, elemValue);
            } else {
                PGX_WARNING("Unsupported operator '%s' in IN expression, defaulting to equality", op.c_str());
                cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                          leftValue, elemValue);
            }
        }

        if (!cmp.getType().isInteger(1)) {
            cmp = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), cmp);
        }

        if (!result) {
            result = cmp;
        } else {
            if (scalar_array_op->useOr) {
                result = ctx.builder.create<mlir::db::OrOp>(ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
                                                            mlir::ValueRange{result, cmp});
            } else {
                result = ctx.builder.create<mlir::db::AndOp>(ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
                                                             mlir::ValueRange{result, cmp});
            }
        }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translate_scalar_array_op_expr OUT: MLIR Value");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr, OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!case_expr) {
        PGX_ERROR("Invalid CaseExpr parameters");
        throw std::runtime_error("Check logs");
    }

    // CASE expressions in PostgreSQL come in two forms:
    // 1. Simple:   CASE expr WHEN val1 THEN result1 WHEN val2 THEN result2 ELSE default END
    // 2. Searched: CASE WHEN cond1 THEN result1 WHEN cond2 THEN result2 ELSE default END
    mlir::Value caseArg = nullptr;
    if (case_expr->arg) {
        caseArg = translate_expression(ctx, case_expr->arg, current_result);
        if (!caseArg) {
            PGX_ERROR("Failed to translate CASE argument expression");
            throw std::runtime_error("Check logs");
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Simple CASE expression with comparison argument");
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Searched CASE expression (no comparison argument)");
    }

    // Build nested if-then-else structure from WHEN clauses
    mlir::Value elseResult = nullptr;
    if (case_expr->defresult) {
        elseResult = translate_expression(ctx, case_expr->defresult, current_result);
        if (!elseResult) {
            PGX_ERROR("Failed to translate CASE ELSE expression");
            throw std::runtime_error("Check logs");
        }
    } else {
        // If no ELSE clause, use NULL as default
        const auto baseType = ctx.builder.getI32Type();
        auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), baseType);
        elseResult = ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), nullableType);
    }

    // Process WHEN clauses in reverse order to build nested if-else chain
    mlir::Value result = elseResult;

    if (case_expr->args && case_expr->args->length > 0) {
        // Process from last to first WHEN clause
        for (int i = case_expr->args->length - 1; i >= 0; i--) {
            const auto whenNode = static_cast<Node*>(lfirst(&case_expr->args->elements[i]));
            if (nodeTag(whenNode) != T_CaseWhen) {
                PGX_ERROR("Expected CaseWhen node in CASE args, got %d", nodeTag(whenNode));
                throw std::runtime_error("Check logs");
            }

            const auto whenClause = reinterpret_cast<CaseWhen*>(whenNode);

            // Translate the WHEN condition
            mlir::Value condition = nullptr;
            if (caseArg) {
                // Simple CASE: whenClause->expr may contain CaseTestExpr that needs to be replaced
                // We need to translate the expression with CaseTestExpr replaced by caseArg
                const mlir::Value whenCondition = translate_expression_with_case_test(ctx, whenClause->expr, caseArg);
                if (!whenCondition) {
                    PGX_ERROR("Failed to translate WHEN condition in simple CASE");
                    throw std::runtime_error("Check logs");
                }
                condition = whenCondition;
            } else {
                condition = translate_expression(ctx, whenClause->expr, current_result);
                if (!condition) {
                    PGX_ERROR("Failed to translate WHEN condition");
                    throw std::runtime_error("Check logs");
                }
            }

            if (auto conditionType = condition.getType();
                !isa<mlir::IntegerType>(conditionType) || cast<mlir::IntegerType>(conditionType).getWidth() != 1)
            {
                condition = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), condition);
            }

            // Translate the THEN result
            mlir::Value thenResult = translate_expression(ctx, whenClause->result, current_result);
            if (!thenResult) {
                PGX_ERROR("Failed to translate THEN result");
                throw std::runtime_error("Check logs");
            }

            // Ensure both branches return the same type
            auto resultType = result.getType();
            // If one is nullable and the other isn't, make both nullable
            if (auto thenType = thenResult.getType(); resultType != thenType) {
                const bool resultIsNullable = isa<mlir::db::NullableType>(resultType);

                if (const bool thenIsNullable = isa<mlir::db::NullableType>(thenType); resultIsNullable && !thenIsNullable)
                {
                    auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), thenType);
                    thenResult = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableType,
                                                                            thenResult);
                } else if (!resultIsNullable && thenIsNullable) {
                    auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), resultType);
                    result = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableType,
                                                                        result);
                    resultType = nullableType;
                }
            }

            // Create if-then-else for this WHEN clause
            auto ifOp = ctx.builder.create<mlir::scf::IfOp>(ctx.builder.getUnknownLoc(), thenResult.getType(),
                                                            condition, true);

            // Build THEN region
            ctx.builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), thenResult);

            // Build ELSE region (contains the previous result)
            ctx.builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), result);

            ctx.builder.setInsertionPointAfter(ifOp);
            result = ifOp.getResult(0);
        }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translate_case_expr OUT: MLIR Value (CASE expression)");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr,
                                                                        const mlir::Value case_test_value)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!expr) {
        throw std::runtime_error("Invalid expression");
    }

    if (expr->type == T_CaseTestExpr) {
        return case_test_value;
    }

    if (expr->type == T_OpExpr) {
        const auto opExpr = reinterpret_cast<OpExpr*>(expr);

        if (!opExpr->args || opExpr->args->length != 2) {
            PGX_ERROR("OpExpr in CASE requires exactly 2 arguments");
            throw std::runtime_error("OpExpr in CASE requires exactly 2 arguments");
        }

        const auto leftNode = static_cast<Node*>(lfirst(&opExpr->args->elements[0]));
        const auto rightNode = static_cast<Node*>(lfirst(&opExpr->args->elements[1]));

        const mlir::Value leftValue = (leftNode && leftNode->type == T_CaseTestExpr)
                                          ? case_test_value
                                          : translate_expression(ctx, reinterpret_cast<Expr*>(leftNode), std::nullopt);
        const mlir::Value rightValue = (rightNode && rightNode->type == T_CaseTestExpr)
                                           ? case_test_value
                                           : translate_expression(ctx, reinterpret_cast<Expr*>(rightNode), std::nullopt);

        if (!leftValue || !rightValue) {
            PGX_ERROR("Failed to translate operands in CASE OpExpr");
            throw std::runtime_error("Check logs");
        }

        return translate_comparison_op(ctx, opExpr->opno, leftValue, rightValue);
    }

    return translate_expression(ctx, expr);
}

auto PostgreSQLASTTranslator::Impl::extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                             OptRefT<const TranslationResult> current_result)
    -> std::optional<std::pair<mlir::Value, mlir::Value>> {
    PGX_IO(AST_TRANSLATE);
    if (!op_expr || !op_expr->args) {
        PGX_ERROR("OpExpr has no arguments");
        throw std::runtime_error("Check logs");
    }

    if (op_expr->args->length < 1) {
        return std::nullopt;
    }

    if (!op_expr->args->elements) {
        PGX_ERROR("OpExpr args list has length %d but no elements array", op_expr->args->length);
        throw std::runtime_error("OpExpr args list has length %d but no elements array");
    }

    mlir::Value lhs;
    mlir::Value rhs;

    for (int argIndex = 0; argIndex < op_expr->args->length && argIndex < 2; argIndex++) {
        const ListCell* lc = &op_expr->args->elements[argIndex];
        if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
            if (const mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode), current_result))
            {
                if (argIndex == 0) {
                    lhs = argValue;
                } else if (argIndex == 1) {
                    rhs = argValue;
                }
            }
        }
    }

    if (!lhs || !rhs) {
        PGX_ERROR("Failed to translate left operand, using placeholder");
        throw std::runtime_error("Failed to translate left operand, using placeholder");
    }

    return std::make_pair(lhs, rhs);
}

auto PostgreSQLASTTranslator::Impl::translate_arithmetic_op(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                            const mlir::Value lhs, const mlir::Value rhs) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!op_expr) {
        PGX_ERROR("Invalid OpExpr");
        throw std::runtime_error("Invalid OpExpr");
    }

    const Oid op_oid = op_expr->opno;
    char* oprname = get_opname(op_oid);
    if (!oprname) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unknown arithmetic operator OID: %d", op_oid);
        throw std::runtime_error("Check logs");
    }

    const std::string op(oprname);
    pfree(oprname);

    if (op != "+" && op != "-" && op != "*" && op != "/" && op != "%") {
        return nullptr;
    }

    const auto loc = ctx.builder.getUnknownLoc();

    // Check if we need to override the result type (for date/interval arithmetic)
    // Check BEFORE upcasting to detect original date/interval types
    auto get_base_type = [](mlir::Type t) -> mlir::Type {
        if (auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return nullable.getType();
        }
        return t;
    };

    const bool has_date_or_interval =
        mlir::isa<mlir::db::DateType>(get_base_type(lhs.getType())) ||
        mlir::isa<mlir::db::DateType>(get_base_type(rhs.getType())) ||
        mlir::isa<mlir::db::IntervalType>(get_base_type(lhs.getType())) ||
        mlir::isa<mlir::db::IntervalType>(get_base_type(rhs.getType()));

    PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC] op=%s, has_date_or_interval=%d, opresulttype=%u",
            op.c_str(), has_date_or_interval, op_expr->opresulttype);

    // Always upcast to ensure compatible operand types
    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, lhs, rhs);

    if (has_date_or_interval) {
        // For date/interval arithmetic, use PostgreSQL's result type
        const PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
        auto result_type = type_mapper.map_postgre_sqltype(op_expr->opresulttype, -1, false);

        PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC DATE] Forcing result type from PostgreSQL opresulttype=%u",
                op_expr->opresulttype);

        // Preserve nullability from operands
        const bool lhs_nullable = mlir::isa<mlir::db::NullableType>(convertedLhs.getType());
        const bool rhs_nullable = mlir::isa<mlir::db::NullableType>(convertedRhs.getType());
        if (lhs_nullable || rhs_nullable) {
            result_type = mlir::db::NullableType::get(ctx.builder.getContext(), result_type);
        }

        if (op == "+")
            return ctx.builder.create<mlir::db::AddOp>(loc, result_type, convertedLhs, convertedRhs);
        if (op == "-")
            return ctx.builder.create<mlir::db::SubOp>(loc, result_type, convertedLhs, convertedRhs);
        if (op == "*")
            return ctx.builder.create<mlir::db::MulOp>(loc, result_type, convertedLhs, convertedRhs);
        if (op == "/")
            return ctx.builder.create<mlir::db::DivOp>(loc, result_type, convertedLhs, convertedRhs);
        if (op == "%")
            return ctx.builder.create<mlir::db::ModOp>(loc, result_type, convertedLhs, convertedRhs);
    } else {
        // For other types, let MLIR infer the result type
        if (op == "+")
            return ctx.builder.create<mlir::db::AddOp>(loc, convertedLhs, convertedRhs);
        if (op == "-")
            return ctx.builder.create<mlir::db::SubOp>(loc, convertedLhs, convertedRhs);
        if (op == "*")
            return ctx.builder.create<mlir::db::MulOp>(loc, convertedLhs, convertedRhs);
        if (op == "/")
            return ctx.builder.create<mlir::db::DivOp>(loc, convertedLhs, convertedRhs);
        if (op == "%")
            return ctx.builder.create<mlir::db::ModOp>(loc, convertedLhs, convertedRhs);
    }

    PGX_ERROR("Failed to create arithmetic operation for operator: %s (OID: %d)", op.c_str(), op_oid);
    throw std::runtime_error("Check logs");
}

struct SQLTypeInference {
    static mlir::FloatType getHigherFloatType(mlir::Type left, mlir::Type right) {
        mlir::FloatType leftFloat = left.dyn_cast_or_null<mlir::FloatType>();
        if (auto rightFloat = right.dyn_cast_or_null<mlir::FloatType>()) {
            if (!leftFloat || rightFloat.getWidth() > leftFloat.getWidth()) {
                return rightFloat;
            }
        }
        return leftFloat;
    }
    static mlir::IntegerType getHigherIntType(mlir::Type left, mlir::Type right) {
        const mlir::IntegerType leftInt = left.dyn_cast_or_null<mlir::IntegerType>();
        if (const auto rightInt = right.dyn_cast_or_null<mlir::IntegerType>()) {
            if (!leftInt || rightInt.getWidth() > leftInt.getWidth()) {
                return rightInt;
            }
        }
        return leftInt;
    }
    static mlir::db::DecimalType getHigherDecimalType(mlir::Type left, mlir::Type right) {
        const auto a = left.dyn_cast_or_null<mlir::db::DecimalType>();
        if (const auto b = right.dyn_cast_or_null<mlir::db::DecimalType>()) {
            if (!a)
                return b;
            const int hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
            const int maxs = std::max(a.getS(), b.getS());
            return mlir::db::DecimalType::get(a.getContext(), std::min(hidig + maxs, MAX_NUMERIC_PRECISION), std::min(maxs, MAX_NUMERIC_UNCONSTRAINED_SCALE));
        }
        return a;
    }
    static mlir::Value castValueToType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
        const bool isNullable = v.getType().isa<mlir::db::NullableType>();
        if (isNullable && !t.isa<mlir::db::NullableType>()) {
            t = mlir::db::NullableType::get(builder.getContext(), t);
        }
        const bool onlyTargetIsNullable = !isNullable && t.isa<mlir::db::NullableType>();
        if (v.getType() == t) {
            return v;
        }
        if (auto* defOp = v.getDefiningOp()) {
            if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
                if (!t.isa<mlir::db::NullableType>()) {
                    constOp.getResult().setType(t);
                    return constOp;
                }
            }
            if (auto nullOp = mlir::dyn_cast_or_null<mlir::db::NullOp>(defOp)) {
                if (nullOp.getResult().getType() == t) { // This was changed from lingodb, unsure if it's going to be
                                                         // problematic...
                    return nullOp;
                }
                return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), t);
            }
        }
        if (v.getType() == getBaseType(t)) {
            return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, v);
        }
        if (onlyTargetIsNullable) {
            mlir::Value casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), getBaseType(t), v);
            return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, casted);
        } else {
            return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
        }
    }
    static mlir::Type getCommonBaseType(mlir::Type left, mlir::Type right) {
        left = getBaseType(left);
        right = getBaseType(right);

        const bool leftIsDate = left.isa<mlir::db::DateType>();
        const bool rightIsDate = right.isa<mlir::db::DateType>();
        const bool leftIsTimestamp = left.isa<mlir::db::TimestampType>();
        const bool rightIsTimestamp = right.isa<mlir::db::TimestampType>();

        if ((leftIsDate || leftIsTimestamp) && (rightIsDate || rightIsTimestamp)) {
            if (leftIsTimestamp)
                return left;
            if (rightIsTimestamp)
                return right;
            return left;
        }

        const bool stringPresent = left.isa<mlir::db::StringType>() || right.isa<mlir::db::StringType>();
        const bool intPresent = left.isa<mlir::IntegerType>() || right.isa<mlir::IntegerType>();
        const bool floatPresent = left.isa<mlir::FloatType>() || right.isa<mlir::FloatType>();
        const bool decimalPresent = left.isa<mlir::db::DecimalType>() || right.isa<mlir::db::DecimalType>();
        if (stringPresent)
            return mlir::db::StringType::get(left.getContext());
        if (decimalPresent)
            return getHigherDecimalType(left, right);
        if (floatPresent)
            return getHigherFloatType(left, right);
        if (intPresent)
            return getHigherIntType(left, right);
        return left;
    }
    static mlir::Type getCommonType(mlir::Type left, mlir::Type right) {
        const bool isNullable = left.isa<mlir::db::NullableType>() || right.isa<mlir::db::NullableType>();
        const auto commonBaseType = getCommonBaseType(left, right);
        if (isNullable) {
            return mlir::db::NullableType::get(left.getContext(), commonBaseType);
        } else {
            return commonBaseType;
        }
    }
    static mlir::Type getCommonBaseType(mlir::TypeRange types) {
        mlir::Type commonType = types.front();
        for (const auto t : types) {
            commonType = getCommonBaseType(commonType, t);
        }
        return commonType;
    }
    static std::vector<mlir::Value> toCommonBaseTypes(mlir::OpBuilder& builder, mlir::ValueRange values) {
        const auto commonType = getCommonBaseType(values.getTypes());
        std::vector<mlir::Value> res;
        for (const auto val : values) {
            res.push_back(castValueToType(builder, val, commonType));
        }
        return res;
    }
    static std::vector<mlir::Value> toCommonBaseTypesExceptDecimals(mlir::OpBuilder& builder, mlir::ValueRange values) {
        std::vector<mlir::Value> res;
        for (auto val : values) {
            if (!getBaseType(val.getType()).isa<mlir::db::DecimalType>()) {
                return toCommonBaseTypes(builder, values);
            }
            res.push_back(val);
        }
        return res;
    }
};

auto PostgreSQLASTTranslator::Impl::upcast_binary_operation(const QueryCtxT& ctx, const mlir::Value lhs,
                                                            const mlir::Value rhs)
    -> std::pair<mlir::Value, mlir::Value> {
    auto convertToType = [&ctx](mlir::Value value, mlir::Type targetBaseType, bool needsNullable) -> mlir::Value {
        const auto currentType = value.getType();
        const auto currentBaseType = getBaseType(currentType);
        const bool isNullable = mlir::isa<mlir::db::NullableType>(currentType);
        const auto loc = ctx.builder.getUnknownLoc();

        if (currentBaseType != targetBaseType) {
            if (isNullable) {
                const auto targetType = mlir::db::NullableType::get(ctx.builder.getContext(), targetBaseType);
                value = ctx.builder.create<mlir::db::CastOp>(loc, targetType, value);
            } else {
                value = ctx.builder.create<mlir::db::CastOp>(loc, targetBaseType, value);
            }
        }

        if (needsNullable && !mlir::isa<mlir::db::NullableType>(value.getType())) {
            const auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), getBaseType(value.getType()));
            value = ctx.builder.create<mlir::db::AsNullableOp>(loc, nullableType, value);
        }

        return value;
    };

    const auto lhsBaseType = getBaseType(lhs.getType());
    const auto rhsBaseType = getBaseType(rhs.getType());
    const auto targetBaseType = (lhsBaseType != rhsBaseType)
                                    ? SQLTypeInference::getCommonBaseType(lhsBaseType, rhsBaseType)
                                    : lhsBaseType;

    const bool needsNullable = mlir::isa<mlir::db::NullableType>(lhs.getType())
                               || mlir::isa<mlir::db::NullableType>(rhs.getType());

    auto convertedLhs = convertToType(lhs, targetBaseType, needsNullable);
    auto convertedRhs = convertToType(rhs, targetBaseType, needsNullable);

    return {convertedLhs, convertedRhs};
}

auto PostgreSQLASTTranslator::Impl::translate_comparison_op(const QueryCtxT& ctx, const Oid op_oid,
                                                            const mlir::Value lhs, const mlir::Value rhs) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!lhs || !rhs) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: nullptr operands for OID %d", op_oid);
        throw std::runtime_error("invalid state");
    }

    char* oprname = get_opname(op_oid);
    if (!oprname) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Failed to get operator name for OID %d", op_oid);
        throw std::runtime_error("invalid state");
    }

    const std::string op(oprname);
    pfree(oprname);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Processing operator '%s' (OID %d)", op.c_str(), op_oid);

    {
        std::string lhsTypeStr, rhsTypeStr;
        llvm::raw_string_ostream lhsOS(lhsTypeStr), rhsOS(rhsTypeStr);
        lhs.getType().print(lhsOS);
        rhs.getType().print(rhsOS);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Comparing types: LHS=%s, RHS=%s", lhsTypeStr.c_str(), rhsTypeStr.c_str());
    }

    mlir::db::DBCmpPredicate predicate;
    if (op == "=") {
        predicate = mlir::db::DBCmpPredicate::eq;
    } else if (op == "<>" || op == "!=") {
        predicate = mlir::db::DBCmpPredicate::neq;
    } else if (op == "<") {
        predicate = mlir::db::DBCmpPredicate::lt;
    } else if (op == "<=") {
        predicate = mlir::db::DBCmpPredicate::lte;
    } else if (op == ">") {
        predicate = mlir::db::DBCmpPredicate::gt;
    } else if (op == ">=") {
        predicate = mlir::db::DBCmpPredicate::gte;
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Unhandled operator: %s (OID: %d)", op.c_str(), op_oid);
        return nullptr;
    }

    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, lhs, rhs);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Creating CmpOp with predicate %d",
            static_cast<int>(predicate));
    return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}

auto PostgreSQLASTTranslator::Impl::translate_expression_for_stream(
    const QueryCtxT& ctx, Expr* expr, mlir::Value input_stream, const std::string& suggested_name,
    const std::vector<TranslationResult::ColumnSchema>& child_columns)
    -> pgx_lower::frontend::sql::StreamExpressionResult {
    PGX_IO(AST_TRANSLATE);

    if (!expr || !input_stream) {
        PGX_ERROR("Invalid parameters for translate_expression_for_stream");
        throw std::runtime_error("Invalid parameters for translate_expression_for_stream");
    }

    auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        PGX_ERROR("RelAlg dialect not registered");
        throw std::runtime_error("RelAlg dialect not registered");
    }
    auto& columnManager = dialect->getColumnManager();

    if (expr->type == T_Var) {
        // We can trivially pass the MLIR value through
        const auto var = reinterpret_cast<Var*>(expr);

        std::string tableName;
        std::string columnName;

        // When child_columns is provided, use it for column resolution (aggregate context)
        // Both OUTER_VAR (-2) and regular vars should use child output positions
        if (var->varattno > 0 && var->varattno <= static_cast<int>(child_columns.size())) {
            const auto& childCol = child_columns[var->varattno - 1];
            tableName = childCol.table_name;
            columnName = childCol.column_name;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Var (varno=%d) resolved to child output column %d: %s.%s", var->varno,
                    var->varattno, tableName.c_str(), columnName.c_str());
        } else {
            throw std::runtime_error("bad situation");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Expression is already a column reference: %s.%s", tableName.c_str(),
                columnName.c_str());

        auto colRef = columnManager.createRef(tableName, columnName);

        auto nested = std::vector{mlir::FlatSymbolRefAttr::get(ctx.builder.getContext(), columnName)};
        auto symbolRef = mlir::SymbolRefAttr::get(ctx.builder.getContext(), tableName, nested);
        auto columnRefAttr = mlir::relalg::ColumnRefAttr::get(ctx.builder.getContext(), symbolRef, colRef.getColumnPtr());

        return {.stream = input_stream, .column_ref = columnRefAttr, .column_name = columnName, .table_name = tableName};
    }

    // For complex expressions, we need to create a MapOp - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    PGX_LOG(AST_TRANSLATE, DEBUG, "Creating MapOp for complex expression (type=%d)", expr->type);
    static size_t exprId = 0;
    const std::string scopeName = "map_expr";
    const std::string columnName = suggested_name.empty() ? "expr_" + std::to_string(exprId++) : suggested_name;

    auto colDef = columnManager.createDef(scopeName, columnName);

    auto tempMapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), input_stream,
                                                             ctx.builder.getArrayAttr({colDef}));

    auto& predicateRegion = tempMapOp.getPredicate();
    auto* block = new mlir::Block;
    predicateRegion.push_back(block);

    auto tupleType = mlir::relalg::TupleType::get(ctx.builder.getContext());
    auto tupleArg = block->addArgument(tupleType, ctx.builder.getUnknownLoc());

    auto blockBuilder = mlir::OpBuilder(ctx.builder.getContext());
    blockBuilder.setInsertionPointToStart(block);

    auto blockCtx = QueryCtxT{ctx.current_stmt, blockBuilder, ctx.current_module, tupleArg, ctx.current_tuple};
    blockCtx.init_plan_results = ctx.init_plan_results;

    TranslationResult childResult;
    childResult.columns = child_columns;

    auto exprValue = translate_expression(blockCtx, expr, childResult);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Finished translating expression");
    if (!exprValue) {
        PGX_ERROR("Failed to translate expression in MapOp");
        throw std::runtime_error("Failed to translate expression in MapOp");
    }

    mlir::Type exprType = exprValue.getType();
    blockBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mlir::ValueRange{exprValue});
    tempMapOp.erase();

    // Create the mapop
    colDef.getColumn().type = exprType;
    auto mapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), input_stream,
                                                         ctx.builder.getArrayAttr({colDef}));
    auto& realRegion = mapOp.getPredicate();
    auto* realBlock = new mlir::Block;
    realRegion.push_back(realBlock);

    auto realTupleArg = realBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

    mlir::OpBuilder realBlockBuilder(ctx.builder.getContext());
    realBlockBuilder.setInsertionPointToStart(realBlock);

    auto realBlockCtx = QueryCtxT{ctx.current_stmt, realBlockBuilder, ctx.current_module, realTupleArg, ctx.current_tuple};
    realBlockCtx.init_plan_results = ctx.init_plan_results;
    auto realExprValue = translate_expression(realBlockCtx, expr, childResult);
    realBlockBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mlir::ValueRange{realExprValue});

    // col ref
    auto nested = std::vector{mlir::FlatSymbolRefAttr::get(ctx.builder.getContext(), columnName)};
    auto symbolRef = mlir::SymbolRefAttr::get(ctx.builder.getContext(), scopeName, nested);
    auto columnRef = mlir::relalg::ColumnRefAttr::get(ctx.builder.getContext(), symbolRef, colDef.getColumnPtr());

    PGX_LOG(AST_TRANSLATE, DEBUG, "Created MapOp with computed column: %s.%s", scopeName.c_str(), columnName.c_str());

    return {.stream = mapOp.getResult(), .column_ref = columnRef, .column_name = columnName, .table_name = scopeName};
}

// ===========================================================================
// Subquery Translation
// ===========================================================================

auto PostgreSQLASTTranslator::Impl::translate_subplan(
    const QueryCtxT& ctx,
    const SubPlan* subplan,
    OptRefT<const TranslationResult> current_result
) -> mlir::Value {
    // Main entry point for SubPlan translation - dispatcher for all subquery types.
    //
    // Overall Steps:
    // 1. Extract subquery Plan from PlannedStmt.subplans using plan_id
    // 2. Call translate_subquery_plan() to get stream + schema
    // 3. Define helper lambdas for common operations:
    //    - build_predicate_block: creates MLIR block for ANY/ALL predicates
    //    - apply_selection_exists: selection + exists pattern (for ANY/ALL)
    // 4. Dispatch on subplan->subLinkType with inline case handlers
    // 5. Return mlir::Value appropriate for each type

    // Step 1: Extract subquery plan from parent statement
    // - subplan->plan_id is 1-indexed into ctx.current_stmt.subplans list
    // - Get Plan* from list_nth(ctx.current_stmt.subplans, plan_id - 1)

    // Step 2: Translate subquery Plan to RelAlg stream
    // auto [subquery_stream, subquery_result] = translate_subquery_plan(ctx, subquery_plan, &ctx.current_stmt);
    // - subquery_stream: mlir::Value of TupleStreamType
    // - subquery_result: TranslationResult with column schema

    // Lambda 1: build_predicate_block
    // Creates MLIR Block for ANY/ALL predicate evaluation
    auto build_predicate_block = [&](bool negate) -> mlir::Block* {
        throw std::runtime_error("UNEXPECTED: Is this possible?");
    };

    // Lambda 2: apply_selection_exists
    // Applies selection with predicate, then wraps result in exists operation
    auto apply_selection_exists = [&](bool negate_predicate) -> mlir::Value {
        throw std::runtime_error("UNEXPECTED: Is this possible?");
    };

    // Step 3: Dispatch on SubLinkType
    switch (subplan->subLinkType) {
        case EXPR_SUBLINK: {
            PGX_LOG(AST_TRANSLATE, DEBUG, "EXPR_SUBLINK: Translating scalar subquery");

            // Extract subquery plan
            if (subplan->plan_id < 1 || subplan->plan_id > list_length(ctx.current_stmt.subplans)) {
                PGX_ERROR("Invalid plan_id=%d (subplans count=%d)", subplan->plan_id,
                          list_length(ctx.current_stmt.subplans));
                throw std::runtime_error("Invalid SubPlan plan_id");
            }

            Plan* subquery_plan = static_cast<Plan*>(list_nth(ctx.current_stmt.subplans, subplan->plan_id - 1));

            // Set up correlation parameters from subplan->parParam and subplan->args
            std::unordered_map<int, std::pair<std::string, std::string>> correlation_mapping;
            if (subplan->parParam && subplan->args) {
                int num_params = list_length(subplan->parParam);
                for (int i = 0; i < num_params; i++) {
                    int param_id = lfirst_int(list_nth_cell(subplan->parParam, i));
                    Expr* arg_expr = static_cast<Expr*>(lfirst(list_nth_cell(subplan->args, i)));

                    if (arg_expr && arg_expr->type == T_Var) {
                        Var* var = reinterpret_cast<Var*>(arg_expr);
                        std::string table_scope = get_table_alias_from_rte(&ctx.current_stmt, var->varno);
                        std::string column_name = get_column_name_from_schema(&ctx.current_stmt, var->varno, var->varattno);

                        correlation_mapping[param_id] = {table_scope, column_name};
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "Mapped correlation paramid=%d to %s.%s",
                                param_id, table_scope.c_str(), column_name.c_str());
                    }
                }
            }

            // Translate subquery with correlation parameters
            auto saved_correlation = ctx.correlation_params;
            const_cast<QueryCtxT&>(ctx).correlation_params = correlation_mapping;
            auto [subquery_stream, subquery_result] = translate_subquery_plan(ctx, subquery_plan, &ctx.current_stmt);
            const_cast<QueryCtxT&>(ctx).correlation_params = saved_correlation;

            if (subquery_result.columns.empty()) {
                PGX_ERROR("Scalar subquery (plan_id=%d) returned no columns", subplan->plan_id);
                throw std::runtime_error("Scalar subquery must return exactly one column");
            }

            // Create GetScalarOp to extract the result
            const auto& result_column = subquery_result.columns[0];
            auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
            auto column_ref = columnManager.createRef(result_column.table_name, result_column.column_name);

            // Make result type nullable (scalar subquery can return NULL)
            mlir::Type result_type = result_column.mlir_type;
            if (!result_type.isa<mlir::db::NullableType>()) {
                result_type = mlir::db::NullableType::get(ctx.builder.getContext(), result_type);
            }

            mlir::Value scalar_value = ctx.builder.create<mlir::relalg::GetScalarOp>(
                ctx.builder.getUnknownLoc(),
                result_type,
                column_ref,
                subquery_stream
            );

            PGX_LOG(AST_TRANSLATE, DEBUG, "EXPR_SUBLINK: Created GetScalarOp for %s.%s",
                    result_column.table_name.c_str(), result_column.column_name.c_str());

            return scalar_value;
        }

        case EXISTS_SUBLINK: {
            throw std::runtime_error("Is this possible?");
        }

        case ANY_SUBLINK: {
            PGX_LOG(AST_TRANSLATE, DEBUG, "ANY_SUBLINK: Translating x IN (subquery) pattern");

            if (subplan->plan_id < 1 || subplan->plan_id > list_length(ctx.current_stmt.subplans)) {
                PGX_ERROR("Invalid SubPlan plan_id: %d", subplan->plan_id);
                throw std::runtime_error("Invalid SubPlan plan_id");
            }

            auto* subquery_plan = static_cast<Plan*>(list_nth(ctx.current_stmt.subplans, subplan->plan_id - 1));
            if (!subquery_plan) {
                PGX_ERROR("SubPlan plan_id %d points to null Plan", subplan->plan_id);
                throw std::runtime_error("Null subquery plan");
            }

            auto [subquery_stream, subquery_result] = translate_subquery_plan(ctx, subquery_plan, &ctx.current_stmt);

            if (!subplan->testexpr) {
                PGX_ERROR("ANY_SUBLINK missing testexpr");
                throw std::runtime_error("ANY_SUBLINK requires testexpr");
            }

            const auto tuple_type = mlir::relalg::TupleType::get(ctx.builder.getContext());

            auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(
                ctx.builder.getUnknownLoc(), subquery_stream);

            auto& pred_region = selection_op.getPredicate();
            auto& pred_block = pred_region.emplaceBlock();
            auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

            mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

            auto inner_ctx = QueryCtxT(ctx.current_stmt, pred_builder, ctx.current_module, inner_tuple, mlir::Value());
            inner_ctx.init_plan_results = ctx.init_plan_results;

            if (subplan->paramIds) {
                const int num_params = list_length(subplan->paramIds);
                PGX_LOG(AST_TRANSLATE, DEBUG, "ANY_SUBLINK: Mapping %d paramIds to subquery columns", num_params);

                for (int i = 0; i < num_params; ++i) {
                    const int param_id = lfirst_int(list_nth_cell(subplan->paramIds, i));

                    if (i < static_cast<int>(subquery_result.columns.size())) {
                        const auto& column_schema = subquery_result.columns[i];
                        pgx_lower::frontend::sql::SubqueryInfo info;
                        info.join_scope = column_schema.table_name;
                        info.join_column_name = column_schema.column_name;
                        info.output_type = column_schema.mlir_type;
                        inner_ctx.subquery_param_mapping[param_id] = info;

                        PGX_LOG(AST_TRANSLATE, DEBUG, "  Mapped paramId=%d to column %s.%s (index %d)",
                                param_id, column_schema.table_name.c_str(), column_schema.column_name.c_str(), i);
                    } else {
                        PGX_ERROR("ParamId=%d index %d exceeds subquery columns (%zu)", param_id, i, subquery_result.columns.size());
                        throw std::runtime_error("ParamId index out of range");
                    }
                }
            }

            auto comparison = translate_expression(inner_ctx, reinterpret_cast<Expr*>(subplan->testexpr), current_result);
            if (!comparison) {
                PGX_ERROR("Failed to translate ANY_SUBLINK testexpr");
                throw std::runtime_error("Failed to translate testexpr");
            }

            pred_builder.create<mlir::relalg::ReturnOp>(
                pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

            auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(
                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), selection_op.getResult());

            PGX_LOG(AST_TRANSLATE, DEBUG, "ANY_SUBLINK: Created EXISTS pattern");
            return exists_op.getResult();
        }

        case ALL_SUBLINK: {
            PGX_LOG(AST_TRANSLATE, DEBUG, "ALL_SUBLINK: Translating price > ALL (subquery) pattern");

            if (subplan->plan_id < 1 || subplan->plan_id > list_length(ctx.current_stmt.subplans)) {
                PGX_ERROR("Invalid SubPlan plan_id: %d", subplan->plan_id);
                throw std::runtime_error("Invalid SubPlan plan_id");
            }

            auto* subquery_plan = static_cast<Plan*>(list_nth(ctx.current_stmt.subplans, subplan->plan_id - 1));
            if (!subquery_plan) {
                PGX_ERROR("SubPlan plan_id %d points to null Plan", subplan->plan_id);
                throw std::runtime_error("Null subquery plan");
            }

            auto [subquery_stream, subquery_result] = translate_subquery_plan(ctx, subquery_plan, &ctx.current_stmt);

            if (!subplan->testexpr) {
                PGX_ERROR("ALL_SUBLINK missing testexpr");
                throw std::runtime_error("ALL_SUBLINK requires testexpr");
            }

            const auto tuple_type = mlir::relalg::TupleType::get(ctx.builder.getContext());

            auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(
                ctx.builder.getUnknownLoc(), subquery_stream);

            auto& pred_region = selection_op.getPredicate();
            auto& pred_block = pred_region.emplaceBlock();
            auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

            mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

            auto inner_ctx = QueryCtxT(ctx.current_stmt, pred_builder, ctx.current_module, inner_tuple, mlir::Value());
            inner_ctx.init_plan_results = ctx.init_plan_results;

            if (subplan->paramIds) {
                const int num_params = list_length(subplan->paramIds);
                PGX_LOG(AST_TRANSLATE, DEBUG, "ALL_SUBLINK: Mapping %d paramIds to subquery columns", num_params);

                for (int i = 0; i < num_params; ++i) {
                    const int param_id = lfirst_int(list_nth_cell(subplan->paramIds, i));

                    if (i < static_cast<int>(subquery_result.columns.size())) {
                        const auto& column_schema = subquery_result.columns[i];
                        pgx_lower::frontend::sql::SubqueryInfo info;
                        info.join_scope = column_schema.table_name;
                        info.join_column_name = column_schema.column_name;
                        info.output_type = column_schema.mlir_type;
                        inner_ctx.subquery_param_mapping[param_id] = info;

                        PGX_LOG(AST_TRANSLATE, DEBUG, "  Mapped paramId=%d to column %s.%s (index %d)",
                                param_id, column_schema.table_name.c_str(), column_schema.column_name.c_str(), i);
                    } else {
                        PGX_ERROR("ParamId=%d index %d exceeds subquery columns (%zu)", param_id, i, subquery_result.columns.size());
                        throw std::runtime_error("ParamId index out of range");
                    }
                }
            }

            auto comparison = translate_expression(inner_ctx, reinterpret_cast<Expr*>(subplan->testexpr), current_result);
            if (!comparison) {
                PGX_ERROR("Failed to translate ALL_SUBLINK testexpr");
                throw std::runtime_error("Failed to translate testexpr");
            }

            auto negated_comparison = pred_builder.create<mlir::db::NotOp>(
                pred_builder.getUnknownLoc(), comparison.getType(), comparison);

            pred_builder.create<mlir::relalg::ReturnOp>(
                pred_builder.getUnknownLoc(), mlir::ValueRange{negated_comparison.getResult()});

            auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(
                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), selection_op.getResult());

            auto final_not = ctx.builder.create<mlir::db::NotOp>(
                ctx.builder.getUnknownLoc(), exists_op.getType(), exists_op.getResult());

            PGX_LOG(AST_TRANSLATE, DEBUG, "ALL_SUBLINK: Created NOT EXISTS pattern");
            return final_not.getResult();
        }

        default: {
            PGX_ERROR("Unsupported SubLinkType: %d", subplan->subLinkType);
            throw std::runtime_error("Unsupported SubLinkType");
        }
    }

    throw std::runtime_error("UNEXPECTED: Is this possible?");
}

auto PostgreSQLASTTranslator::Impl::translate_subquery_plan(
    const QueryCtxT& parent_ctx,
    Plan* subquery_plan,
    const PlannedStmt* parent_stmt
) -> std::pair<mlir::Value, TranslationResult> {
    // Translate subquery Plan tree to RelAlg MLIR stream with isolated context.
    // This function provides the foundation for all subquery types.
    //
    // Steps:
    // 1. Create new QueryCtxT for subquery:
    //    - Copy parent_ctx but with fresh context for subquery scope
    //    - Keep same builder, module, stmt reference
    //    - current_tuple: use parent current_tuple for correlation support
    // 2. Call translate_plan_node() recursively:
    //    - Pass subquery_plan as the plan to translate
    //    - This reuses ALL existing plan translation infrastructure
    //    - Returns TranslationResult with op and column schema
    // 3. Extract result stream from TranslationResult:
    //    - Get mlir::Value from result.op->getResult(0)
    //    - Verify it's TupleStreamType (not materialized table)
    // 4. Return pair:
    //    - First: mlir::Value of TupleStreamType
    //    - Second: TranslationResult with full column schema
    // 5. Context automatically destroyed on return (RAII pattern)
    //
    // Note: This function is the key reuse point - it lets us treat subqueries
    // as just another Plan tree that we already know how to translate.

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_subquery_plan: Starting subquery translation");

    auto subquery_ctx = QueryCtxT(
        *parent_stmt,
        parent_ctx.builder,
        parent_ctx.current_module,
        parent_ctx.current_tuple,
        mlir::Value()
    );
    subquery_ctx.init_plan_results = parent_ctx.init_plan_results;
    subquery_ctx.correlation_params = parent_ctx.correlation_params;

    auto subquery_result = translate_plan_node(subquery_ctx, subquery_plan);

    if (!subquery_result.op) {
        PGX_ERROR("translate_subquery_plan: SubPlan translation returned null operation");
        throw std::runtime_error("Subquery translation failed");
    }

    mlir::Value subquery_stream = subquery_result.op->getResult(0);

    if (!subquery_stream) {
        PGX_ERROR("translate_subquery_plan: SubPlan operation has no result stream");
        throw std::runtime_error("Subquery has no stream result");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_subquery_plan: Successfully translated subquery with %zu columns",
            subquery_result.columns.size());

    return {subquery_stream, subquery_result};
}

auto PostgreSQLASTTranslator::Impl::translate_param(
    const QueryCtxT& ctx,
    const Param* param,
    OptRefT<const TranslationResult> current_result
) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!param) {
        PGX_ERROR("Invalid Param node");
        throw std::runtime_error("Invalid Param node");
    }

    if (param->paramkind != PARAM_EXEC) {
        PGX_ERROR("Only PARAM_EXEC parameters are supported (got paramkind=%d)", param->paramkind);
        throw std::runtime_error("Unsupported param kind");
    }

    // Check correlation parameters first (these are free variables and don't need current_result)
    const auto& correlation_params = ctx.correlation_params;
    const auto corr_it = correlation_params.find(param->paramid);

    if (corr_it != correlation_params.end()) {
        const auto& [tableScope, columnName] = corr_it->second;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving Param paramid=%d to correlation parameter %s.%s as free variable",
                param->paramid, tableScope.c_str(), columnName.c_str());

        auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        auto column_ref = columnManager.createRef(tableScope, columnName);

        const auto type_mapper = PostgreSQLTypeMapper(context_);
        auto column_type = type_mapper.map_postgre_sqltype(param->paramtype, param->paramtypmod, true);

        // Use outer_tuple for correlation parameters (free variables)
        mlir::Value tuple_to_use = ctx.outer_tuple ? ctx.outer_tuple : ctx.current_tuple;
        return ctx.builder.create<mlir::relalg::GetColumnOp>(
            ctx.builder.getUnknownLoc(),
            column_type,
            column_ref,
            tuple_to_use
        );
    }

    const auto& subquery_mapping = ctx.subquery_param_mapping;
    const auto subquery_it = subquery_mapping.find(param->paramid);

    if (subquery_it != subquery_mapping.end()) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving Param paramid=%d to subquery column", param->paramid);

        const auto& [join_scope, join_column_name, output_type] = subquery_it->second;

        auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        auto column_ref = columnManager.createRef(join_scope, join_column_name);

        mlir::Value column_value = ctx.builder.create<mlir::relalg::GetColumnOp>(
            ctx.builder.getUnknownLoc(),
            output_type,
            column_ref,
            ctx.current_tuple
        );

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created GetColumnOp for Param paramid=%d from subquery tuple column %s.%s",
                param->paramid, join_scope.c_str(), join_column_name.c_str());

        return column_value;
    }

    // If we reach here, this must be an InitPlan parameter - look it up in context
    const auto& init_plan_results = ctx.init_plan_results;
    const auto it = init_plan_results.find(param->paramid);

    if (it == init_plan_results.end()) {
        PGX_ERROR("Param references unknown paramid=%d (InitPlan not processed?)", param->paramid);
        throw std::runtime_error("Param references unknown InitPlan result");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving Param paramid=%d to InitPlan result", param->paramid);

    // Extract scalar value from InitPlan result using GetScalarOp
    const auto& initplan_result = it->second;

    if (!initplan_result.op) {
        PGX_ERROR("InitPlan result for paramid=%d has no operation", param->paramid);
        throw std::runtime_error("Invalid InitPlan result");
    }

    if (initplan_result.columns.empty()) {
        PGX_ERROR("InitPlan result for paramid=%d has no columns", param->paramid);
        throw std::runtime_error("InitPlan must return at least one column");
    }

    mlir::Value stream = initplan_result.op->getResult(0);
    const auto& first_column = initplan_result.columns[0];

    auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    auto column_ref = columnManager.createRef(first_column.table_name, first_column.column_name);

    mlir::Value scalar_value = ctx.builder.create<mlir::relalg::GetScalarOp>(
        ctx.builder.getUnknownLoc(),
        first_column.mlir_type,
        column_ref,
        stream
    );

    PGX_LOG(AST_TRANSLATE, DEBUG, "Created GetScalarOp for Param paramid=%d from %s.%s",
            param->paramid, first_column.table_name.c_str(), first_column.column_name.c_str());

    return scalar_value;
}

} // namespace postgresql_ast