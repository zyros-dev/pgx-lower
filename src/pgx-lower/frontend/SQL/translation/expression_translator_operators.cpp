#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "nodes/nodeFuncs.h"
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

    auto [lhs, rhs] = *operands;
    std::tie(lhs, rhs) = normalize_bpchar_operands(ctx, op_expr, lhs, rhs);
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
    case AND_EXPR: {
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

    case OR_EXPR: {
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

    case NOT_EXPR: {
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

auto PostgreSQLASTTranslator::Impl::translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr,
                                                        OptRefT<const TranslationResult> current_result) -> mlir::Value {
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

auto PostgreSQLASTTranslator::Impl::translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr,
                                                            OptRefT<const TranslationResult> current_result)
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
    auto leftValue = translate_expression(ctx, reinterpret_cast<Expr*>(leftNode), current_result);
    if (!leftValue) {
        PGX_ERROR("Failed to translate left operand of IN expression");
        throw std::runtime_error("Failed to translate left operand of IN expression");
    }

    Oid leftTypeOid = exprType(leftNode);
    int32 leftTypeMod = exprTypmod(leftNode);
    int bpcharLength = -1;
    if (leftTypeOid == BPCHAROID && leftTypeMod >= VARHDRSZ) {
        bpcharLength = leftTypeMod - VARHDRSZ;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Left operand is BPCHAR with length=%d", bpcharLength);
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
        } else if (constNode->consttype == BPCHARARRAYOID) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing BPCHAR array (CHAR/VARCHAR), target column length=%d",
                    bpcharLength);
            const auto array = DatumGetArrayTypeP(constNode->constvalue);
            int nitems;
            Datum* values;
            bool* nulls;

            deconstruct_array(array, BPCHAROID, -1, false, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    const auto bpcharValue = DatumGetBpCharP(values[i]);
                    std::string str_value(VARDATA_ANY(bpcharValue), VARSIZE_ANY_EXHDR(bpcharValue));

                    str_value.erase(str_value.find_last_not_of(' ') + 1);

                    if (bpcharLength > 0 && str_value.length() < static_cast<size_t>(bpcharLength)) {
                        str_value.resize(bpcharLength, ' ');
                        PGX_LOG(AST_TRANSLATE, DEBUG, "BPCHAR array element[%d]: '%s' (padded to len=%d)", i,
                                str_value.c_str(), bpcharLength);
                    } else {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "BPCHAR array element[%d]: '%s' (len=%zu, no padding needed)", i,
                                str_value.c_str(), str_value.length());
                    }

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

        auto& col_mgr = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

        const auto tuple_type = mlir::relalg::TupleType::get(ctx.builder.getContext());

        auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), initplan_stream);

        auto& pred_region = selection_op.getPredicate();
        auto& pred_block = pred_region.emplaceBlock();
        auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

        mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

        auto initplan_col_ref = col_mgr.createRef(initplan_column.table_name, initplan_column.column_name);
        auto initplan_value = pred_builder.create<mlir::relalg::GetColumnOp>(
            pred_builder.getUnknownLoc(), initplan_column.mlir_type, initplan_col_ref, inner_tuple);

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

        auto comparison = pred_builder.create<mlir::db::CmpOp>(pred_builder.getUnknownLoc(), predicate, leftValue,
                                                               initplan_value);

        pred_builder.create<mlir::relalg::ReturnOp>(pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

        auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(ctx.builder.getUnknownLoc(),
                                                                    ctx.builder.getI1Type(), selection_op.getResult());

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created EXISTS pattern for ScalarArrayOpExpr with Param");
        return exists_op.getResult();
    } else {
        PGX_ERROR("ScalarArrayOpExpr: Unexpected right operand type %d", nodeTag(rightNode));
        throw std::runtime_error("Unsupported ScalarArrayOpExpr operand type");
    }

    if (arrayElements.empty()) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Empty array in IN clause, returning %s",
                scalar_array_op->useOr ? "false" : "true");
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                              scalar_array_op->useOr ? 0 : 1, ctx.builder.getI1Type());
    }

    char* oprname = get_opname(scalar_array_op->opno);
    std::string op = oprname ? std::string(oprname) : "=";
    if (oprname)
        pfree(oprname);

    if (op == "=" && scalar_array_op->useOr) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using db.oneof for IN clause with %zu array elements", arrayElements.size());

        std::vector<mlir::Value> values;
        values.push_back(leftValue);
        values.insert(values.end(), arrayElements.begin(), arrayElements.end());

        auto oneofOp = ctx.builder.create<mlir::db::OneOfOp>(ctx.builder.getUnknownLoc(), values);
        PGX_LOG(AST_TRANSLATE, IO, "translate_scalar_array_op_expr OUT: db.oneof MLIR Value");
        return oneofOp.getResult();
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Using comparison loop for operator '%s' with useOr=%d", op.c_str(),
            scalar_array_op->useOr);

    mlir::Value result = nullptr;
    for (const auto& elemValue : arrayElements) {
        mlir::Value cmp = nullptr;

        if (op == "=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      leftValue, elemValue);
        } else if (op == "<>" || op == "!=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::neq,
                                                      leftValue, elemValue);
        } else {
            PGX_WARNING("Unsupported operator '%s' in ScalarArrayOpExpr, defaulting to equality", op.c_str());
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      leftValue, elemValue);
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

auto PostgreSQLASTTranslator::Impl::normalize_bpchar_operands(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                              mlir::Value lhs, mlir::Value rhs)
    -> std::pair<mlir::Value, mlir::Value> {
    if (!op_expr || !op_expr->args || op_expr->args->length != 2) {
        return {lhs, rhs};
    }

    auto get_base_type = [](mlir::Type t) -> mlir::Type {
        if (auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return nullable.getType();
        }
        return t;
    };

    const bool lhs_is_string = mlir::isa<mlir::db::StringType>(get_base_type(lhs.getType()));
    const bool rhs_is_string = mlir::isa<mlir::db::StringType>(get_base_type(rhs.getType()));

    if (!lhs_is_string || !rhs_is_string) {
        return {lhs, rhs};
    }

    auto* lhs_expr = reinterpret_cast<Expr*>(lfirst(&op_expr->args->elements[0]));
    auto* rhs_expr = reinterpret_cast<Expr*>(lfirst(&op_expr->args->elements[1]));

    auto extract_bpchar_length = [](Expr* expr) -> int {
        if (!expr) {
            return -1;
        }
        const Oid typeOid = exprType(reinterpret_cast<Node*>(expr));
        const int32 typeMod = exprTypmod(reinterpret_cast<Node*>(expr));

        if (typeOid == BPCHAROID && typeMod >= VARHDRSZ) {
            return typeMod - VARHDRSZ;
        }
        return -1;
    };

    auto pad_string_constant = [&](mlir::Value val, int target_length) -> mlir::Value {
        auto* defOp = val.getDefiningOp();
        if (!defOp || !mlir::isa<mlir::db::ConstantOp>(defOp)) {
            return val;
        }

        auto constOp = mlir::cast<mlir::db::ConstantOp>(defOp);
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constOp.getValue())) {
            std::string str_value = strAttr.getValue().str();
            if (static_cast<int>(str_value.length()) < target_length) {
                str_value.resize(target_length, ' ');
                PGX_LOG(AST_TRANSLATE, DEBUG, "Padded BPCHAR constant to length %d: '%s'", target_length,
                        str_value.c_str());

                return ctx.builder.create<mlir::db::ConstantOp>(ctx.builder.getUnknownLoc(),
                                                                ctx.builder.getType<mlir::db::StringType>(),
                                                                ctx.builder.getStringAttr(str_value));
            }
        }
        return val;
    };

    const int lhs_bpchar_len = extract_bpchar_length(lhs_expr);
    const int rhs_bpchar_len = extract_bpchar_length(rhs_expr);

    if (lhs_bpchar_len > 0 && rhs_expr->type == T_Const) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Normalizing RHS constant to match LHS BPCHAR(%d)", lhs_bpchar_len);
        rhs = pad_string_constant(rhs, lhs_bpchar_len);
    } else if (rhs_bpchar_len > 0 && lhs_expr->type == T_Const) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Normalizing LHS constant to match RHS BPCHAR(%d)", rhs_bpchar_len);
        lhs = pad_string_constant(lhs, rhs_bpchar_len);
    }

    return {lhs, rhs};
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
    auto get_base_type = [](mlir::Type t) -> mlir::Type {
        if (auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return nullable.getType();
        }
        return t;
    };

    const bool has_date_or_interval = mlir::isa<mlir::db::DateType>(get_base_type(lhs.getType()))
                                      || mlir::isa<mlir::db::DateType>(get_base_type(rhs.getType()))
                                      || mlir::isa<mlir::db::IntervalType>(get_base_type(lhs.getType()))
                                      || mlir::isa<mlir::db::IntervalType>(get_base_type(rhs.getType()));

    PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC] op=%s, has_date_or_interval=%d, opresulttype=%u", op.c_str(),
            has_date_or_interval, op_expr->opresulttype);

    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, lhs, rhs);

    if (has_date_or_interval) {
        const PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
        auto result_type = type_mapper.map_postgre_sqltype(op_expr->opresulttype, -1, false);

        PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC DATE] Forcing result type from PostgreSQL opresulttype=%u",
                op_expr->opresulttype);

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
        mlir::SmallVector<mlir::Type, 1> inferredTypes;

        if (op == "+") {
            if (mlir::failed(mlir::db::AddOp::inferReturnTypes(
                    ctx.builder.getContext(), loc, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferredTypes)))
            {
                PGX_ERROR("Failed to infer AddOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::AddOp>(loc, inferredTypes[0], convertedLhs, convertedRhs);
        }
        if (op == "-") {
            if (mlir::failed(mlir::db::SubOp::inferReturnTypes(
                    ctx.builder.getContext(), loc, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferredTypes)))
            {
                PGX_ERROR("Failed to infer SubOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::SubOp>(loc, inferredTypes[0], convertedLhs, convertedRhs);
        }
        if (op == "*") {
            if (mlir::failed(mlir::db::MulOp::inferReturnTypes(
                    ctx.builder.getContext(), loc, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferredTypes)))
            {
                PGX_ERROR("Failed to infer MulOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::MulOp>(loc, inferredTypes[0], convertedLhs, convertedRhs);
        }
        if (op == "/") {
            if (mlir::failed(mlir::db::DivOp::inferReturnTypes(
                    ctx.builder.getContext(), loc, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferredTypes)))
            {
                PGX_ERROR("Failed to infer DivOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::DivOp>(loc, inferredTypes[0], convertedLhs, convertedRhs);
        }
        if (op == "%") {
            if (mlir::failed(mlir::db::ModOp::inferReturnTypes(
                    ctx.builder.getContext(), loc, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferredTypes)))
            {
                PGX_ERROR("Failed to infer ModOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::ModOp>(loc, inferredTypes[0], convertedLhs, convertedRhs);
        }
    }

    PGX_ERROR("Failed to create arithmetic operation for operator: %s (OID: %d)", op.c_str(), op_oid);
    throw std::runtime_error("Check logs");
}

struct SQLTypeInference {
    static mlir::FloatType getHigherFloatType(mlir::Type left, mlir::Type right) {
        auto leftFloat = dyn_cast_or_null<mlir::FloatType>(left);
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
            return mlir::db::DecimalType::get(a.getContext(), std::min(hidig + maxs, MAX_NUMERIC_PRECISION),
                                              std::min(maxs, MAX_NUMERIC_UNCONSTRAINED_SCALE));
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
} // namespace postgresql_ast