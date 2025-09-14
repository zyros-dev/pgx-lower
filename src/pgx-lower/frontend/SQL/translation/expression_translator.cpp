#include "translator_internals.h"
#include "pgx-lower/utility/stacktrace.h"
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
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
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

auto PostgreSQLASTTranslator::Impl::translate_expression(const QueryCtxT& ctx, Expr* expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!expr) {
        PGX_ERROR("Expression is null");
        return nullptr;
    }

    switch (expr->type) {
    case T_Var:
    case LINGODB_T_VAR: return translate_var(ctx, reinterpret_cast<Var*>(expr));
    case T_Const: return translate_const(ctx, reinterpret_cast<Const*>(expr));
    case T_OpExpr:
    case LINGODB_T_OPEXPR: return translate_op_expr(ctx, reinterpret_cast<OpExpr*>(expr));
    case T_FuncExpr: return translate_func_expr(ctx, reinterpret_cast<FuncExpr*>(expr));
    case T_BoolExpr: return translate_bool_expr(ctx, reinterpret_cast<BoolExpr*>(expr));
    case T_Aggref: return translate_aggref(ctx, reinterpret_cast<Aggref*>(expr));
    case T_NullTest: return translate_null_test(ctx, reinterpret_cast<NullTest*>(expr));
    case T_CoalesceExpr: return translate_coalesce_expr(ctx, reinterpret_cast<CoalesceExpr*>(expr));
    case T_ScalarArrayOpExpr: return translate_scalar_array_op_expr(ctx, reinterpret_cast<ScalarArrayOpExpr*>(expr));
    case T_CaseExpr: return translate_case_expr(ctx, reinterpret_cast<CaseExpr*>(expr));
    case T_CaseTestExpr: {
        PGX_WARNING("CaseTestExpr encountered outside of CASE expression context");
        return nullptr;
    }
    case T_RelabelType: {
        const auto* relabel = reinterpret_cast<RelabelType*>(expr);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unwrapping T_RelabelType to translate underlying expression");
        return translate_expression(ctx, relabel->arg);
    }
    default: {
        PGX_ERROR("Unsupported expression type: %d", expr->type);
        throw std::runtime_error("Unsupported expression type - read the logs");
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!op_expr) {
        PGX_ERROR("Invalid OpExpr parameters");
        return nullptr;
    }

    auto operands = extract_op_expr_operands(ctx, op_expr);
    if (!operands) {
        PGX_ERROR("Failed to extract OpExpr operands");
        return nullptr;
    }

    auto [lhs, rhs] = *operands;
    const Oid opOid = op_expr->opno;

    mlir::Value result = translate_arithmetic_op(ctx, opOid, lhs, rhs);
    if (result) {
        return result;
    }

    result = translate_comparison_op(ctx, opOid, lhs, rhs);
    if (result) {
        return result;
    }

    switch (opOid) {
    case PG_TEXT_LIKE_OID: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LIKE operator to db.runtime_call");

        auto convertedLhs = lhs;
        auto convertedRhs = rhs;

        const auto lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
        const auto rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

        if (lhsNullable && !rhsNullable) {
            auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
            convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType, rhs);
        }
        else if (!lhsNullable && rhsNullable) {
            auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
            convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType, lhs);
        }

        const bool hasNullableOperand = lhsNullable || rhsNullable;
        auto resultType = hasNullableOperand
                              ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), ctx.builder.getI1Type()))
                              : mlir::Type(ctx.builder.getI1Type());

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(),
                                                            resultType,
                                                            ctx.builder.getStringAttr("Like"),
                                                            mlir::ValueRange{convertedLhs, convertedRhs});

        return op.getRes();
    }
    case PG_TEXT_NOT_LIKE_OID: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NOT LIKE operator to negated db.runtime_call");
        auto convertedLhs = lhs;
        auto convertedRhs = rhs;

        const bool lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
        const bool rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

        if (lhsNullable && !rhsNullable) {
            auto nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
            convertedRhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType, rhs);
        }
        else if (!lhsNullable && rhsNullable) {
            auto nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
            convertedLhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType, lhs);
        }

        // Create the LIKE operation
        const mlir::Type boolType = ctx.builder.getI1Type();
        auto resultType = (lhsNullable || rhsNullable)
                              ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(), boolType))
                              : boolType;

        auto likeOp = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(),
                                                                resultType,
                                                                ctx.builder.getStringAttr("Like"),
                                                                mlir::ValueRange{convertedLhs, convertedRhs});

        // Negate the result using NotOp
        auto notOp = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), resultType, likeOp.getRes());

        return notOp.getResult();
    }
    case PG_TEXT_CONCAT_OID: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating || operator to StringRuntime::concat");

        const bool hasNullableOperand = isa<mlir::db::NullableType>(lhs.getType())
                                        || isa<mlir::db::NullableType>(rhs.getType());

        auto resultType =
            hasNullableOperand
                ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                         mlir::db::StringType::get(ctx.builder.getContext())))
                : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(),
                                                            resultType,
                                                            ctx.builder.getStringAttr("Concat"),
                                                            mlir::ValueRange{lhs, rhs});

        return op.getRes();
    }
    default: PGX_ERROR("Unsupported operator OID: %d", opOid); throw std::runtime_error("Unsupported operator");
    }
}

auto PostgreSQLASTTranslator::Impl::translate_var(const QueryCtxT& ctx, const Var* var) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!var || !ctx.current_tuple) {
        const auto trace = pgx_lower::utility::capture_stacktrace();
        PGX_ERROR("Invalid Var parameters: var=%p, builder=%p, tuple=%p\n%s",
                  var,
                  ctx.builder,
                  ctx.current_tuple.getAsOpaquePointer(),
                  trace.c_str());
        throw std::runtime_error("Invalid Var parameters");
    }

    // For RelAlg operations, we need to generate a GetColumnOp
    // This requires the current tuple value and column reference

    if (ctx.current_tuple) {
        // We have a tuple handle - use it to get the column value
        // This would typically be inside a MapOp or SelectionOp region

        // Get real table and column names from PostgreSQL schema
        const auto tableName = get_table_name_from_rte(&ctx.current_stmt, var->varno);
        const auto colName = get_column_name_from_schema(&ctx.current_stmt, var->varno, var->varattno);

        // Get column manager from RelAlg dialect
        auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        if (!dialect) {
            PGX_ERROR("RelAlg dialect not registered");
            return nullptr;
        }

        auto& columnManager = dialect->getColumnManager();

        // Map PostgreSQL type to MLIR type - check if column is nullable
        PostgreSQLTypeMapper type_mapper(context_);
        const bool nullable = is_column_nullable(&ctx.current_stmt, var->varno, var->varattno);
        auto mlirType = type_mapper.map_postgre_sqltype(var->vartype, var->vartypmod, nullable);

        // This ensures proper column tracking and avoids invalid attributes
        auto colRef = columnManager.createRef(tableName, colName);

        // Set the column type
        colRef.getColumn().type = mlirType;

        auto getColOp =
            ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), mlirType, colRef, ctx.current_tuple);

        return getColOp.getRes();
    }
    else {
        // No tuple context - this shouldn't happen in properly structured queries
        PGX_WARNING("No tuple context for Var translation, using placeholder");
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                              DEFAULT_PLACEHOLDER_INT,
                                                              ctx.builder.getI32Type());
    }
}

auto PostgreSQLASTTranslator::Impl::translate_const(const QueryCtxT& ctx, Const* const_node) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    return postgresql_ast::translate_const(const_node, ctx.builder, context_);
}

auto PostgreSQLASTTranslator::Impl::translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!func_expr) {
        PGX_ERROR("Invalid FuncExpr parameters");
        return nullptr;
    }

    // Translate function arguments first
    auto args = std::vector<mlir::Value>{};
    if (func_expr->args && func_expr->args->length > 0) {
        // Safety check for elements array (PostgreSQL 17)
        if (!func_expr->args->elements) {
            PGX_WARNING("FuncExpr args list has length but no elements array");
            return nullptr;
        }

        // Iterate through arguments
        for (int i = 0; i < func_expr->args->length; i++) {
            const ListCell* lc = &func_expr->args->elements[i];
            if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
                    args.push_back(argValue);
                }
            }
        }
    }

    // Map PostgreSQL function OID to MLIR operations

    const auto loc = ctx.builder.getUnknownLoc();

    switch (func_expr->funcid) {
    case PG_F_ABS_INT4:
    case PG_F_ABS_INT8:
    case PG_F_ABS_FLOAT4:
    case PG_F_ABS_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("ABS requires exactly 1 argument, got %d", args.size());
            return nullptr;
        }
        // Implement absolute value using comparison and negation
        // Since DB dialect doesn't have AbsOp, use arith operations
        {
            auto zero = ctx.builder.create<mlir::arith::ConstantIntOp>(loc, DEFAULT_PLACEHOLDER_INT, args[0].getType());
            auto cmp = ctx.builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, args[0], zero);
            auto neg = ctx.builder.create<mlir::arith::SubIOp>(loc, zero, args[0]);
            return ctx.builder.create<mlir::arith::SelectOp>(loc, cmp, neg, args[0]);
        }

    case PG_F_SQRT_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("SQRT requires exactly 1 argument");
            return nullptr;
        }
        // Use math dialect sqrt (TODO: may need to add math dialect)
        PGX_WARNING("SQRT function not yet implemented in DB dialect");
        return args[0];

    case PG_F_POW_FLOAT8:
        if (args.size() != 2) {
            PGX_ERROR("POWER requires exactly 2 arguments");
            return nullptr;
        }
        PGX_WARNING("POWER function not yet implemented in DB dialect");
        return args[0];

    case PG_F_UPPER: {
        if (args.size() != 1) {
            PGX_ERROR("UPPER requires exactly 1 argument");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating UPPER function to StringRuntime::upper");

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        mlir::Type resultType =
            hasNullableOperand
                ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                         mlir::db::StringType::get(ctx.builder.getContext())))
                : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc,
                                                            resultType,
                                                            ctx.builder.getStringAttr("Upper"),
                                                            mlir::ValueRange{args[0]});
        return op.getRes();
    }

    case PG_F_LOWER: {
        if (args.size() != 1) {
            PGX_ERROR("LOWER requires exactly 1 argument");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LOWER function to StringRuntime::lower");

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        mlir::Type resultType =
            hasNullableOperand
                ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                         mlir::db::StringType::get(ctx.builder.getContext())))
                : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc,
                                                            resultType,
                                                            ctx.builder.getStringAttr("Lower"),
                                                            mlir::ValueRange{args[0]});
        return op.getRes();
    }

    case PG_F_SUBSTRING: {
        if (args.size() < 2 || args.size() > 3) {
            PGX_ERROR("SUBSTRING requires 2 or 3 arguments, got %d", args.size());
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating SUBSTRING function to StringRuntime::substring");

        // SUBSTRING(string, start [, length])
        // If length is not provided, we need to add a default
        auto substringArgs = std::vector{args[0], args[1]};
        if (args.size() == 3) {
            substringArgs.push_back(args[2]);
        }
        else {
            // Default length to max int32 for "rest of string"
            auto maxLength = ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 2147483647, ctx.builder.getI32Type());
            substringArgs.push_back(maxLength);
        }

        const bool hasNullableOperand = isa<mlir::db::NullableType>(args[0].getType());
        mlir::Type resultType =
            hasNullableOperand
                ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                         mlir::db::StringType::get(ctx.builder.getContext())))
                : mlir::Type(mlir::db::StringType::get(ctx.builder.getContext()));

        auto op = ctx.builder.create<mlir::db::RuntimeCall>(loc,
                                                            resultType,
                                                            ctx.builder.getStringAttr("Substring"),
                                                            mlir::ValueRange{substringArgs});
        return op.getRes();
    }

    case PG_F_LENGTH:
        if (args.size() != 1) {
            PGX_ERROR("LENGTH requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("LENGTH function not yet implemented");
        return ctx.builder.create<mlir::arith::ConstantIntOp>(loc, DEFAULT_PLACEHOLDER_INT, ctx.builder.getI32Type());

    case PG_F_CEIL_FLOAT8:
    case PG_F_FLOOR_FLOAT8:
    case PG_F_ROUND_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("Rounding function requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("Rounding functions not yet implemented in DB dialect");
        return args[0];

    default: {
        PGX_WARNING("Unknown function OID %d", func_expr->funcid);
        throw std::runtime_error("Unknown function OID " + std::to_string(func_expr->funcid));
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!bool_expr) {
        PGX_ERROR("Invalid BoolExpr parameters");
        return nullptr;
    }

    if (!bool_expr->args || bool_expr->args->length == 0) {
        PGX_ERROR("BoolExpr has no arguments");
        return nullptr;
    }

    switch (bool_expr->boolop) {
    case BOOL_AND_EXPR: {
        mlir::Value result = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            if (!bool_expr->args->elements) {
                PGX_WARNING("BoolExpr AND args list has length but no elements array");
                return nullptr;
            }

            for (int i = 0; i < bool_expr->args->length; i++) {
                const ListCell* lc = &bool_expr->args->elements[i];
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
                        // Ensure boolean type
                        if (!argValue.getType().isInteger(1)) {
                            argValue = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        }
                        else {
                            result = ctx.builder.create<mlir::db::AndOp>(ctx.builder.getUnknownLoc(),
                                                                         ctx.builder.getI1Type(),
                                                                         mlir::ValueRange{result, argValue});
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
            // Safety check for elements array (PostgreSQL 17)
            if (!bool_expr->args->elements) {
                PGX_WARNING("BoolExpr OR args list has length but no elements array");
                return nullptr;
            }

            // Iterate using PostgreSQL 17 style with elements array
            for (int i = 0; i < bool_expr->args->length; i++) {
                const ListCell* lc = &bool_expr->args->elements[i];
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
                        // Ensure boolean type
                        if (!argValue.getType().isInteger(1)) {
                            argValue = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        }
                        else {
                            result = ctx.builder.create<mlir::db::OrOp>(ctx.builder.getUnknownLoc(),
                                                                        ctx.builder.getI1Type(),
                                                                        mlir::ValueRange{result, argValue});
                        }
                    }
                }
            }
        }

        if (!result) {
            // Default to false if no valid expression
            result = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                                    DEFAULT_PLACEHOLDER_BOOL_FALSE,
                                                                    ctx.builder.getI1Type());
        }
        return result;
    }

    case BOOL_NOT_EXPR: {
        // NOT has single argument
        mlir::Value argVal = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            // Get first argument
            if (const ListCell* lc = list_head(bool_expr->args)) {
                if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                    argVal = translate_expression(ctx, reinterpret_cast<Expr*>(argNode));
                }
            }
        }

        if (!argVal) {
            // Default argument if none provided
            PGX_WARNING("NOT expression has no valid argument, using placeholder");
            argVal = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                                    DEFAULT_PLACEHOLDER_BOOL,
                                                                    ctx.builder.getI1Type());
        }

        // Ensure argument is boolean
        if (!argVal.getType().isInteger(1)) {
            argVal = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), argVal);
        }

        return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), argVal);
    }

    default: PGX_ERROR("Unknown BoolExpr type: %d", bool_expr->boolop); return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translate_null_test(const QueryCtxT& ctx, const NullTest* null_test) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!null_test) {
        PGX_ERROR("Invalid NullTest parameters");
        return nullptr;
    }

    // Translate the argument expression
    auto* argNode = reinterpret_cast<Node*>(null_test->arg);
    auto argVal = translate_expression(ctx, reinterpret_cast<Expr*>(argNode));
    if (!argVal) {
        PGX_ERROR("Failed to translate NullTest argument");
        return nullptr;
    }

    // Follow LingoDB's exact pattern: check if type is nullable first
    if (isa<mlir::db::NullableType>(argVal.getType())) {
        mlir::Value isNull = ctx.builder.create<mlir::db::IsNullOp>(ctx.builder.getUnknownLoc(), argVal);
        if (null_test->nulltesttype == PG_IS_NOT_NULL) {
            // LingoDB's clean approach: use NotOp instead of XOrIOp
            return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), isNull);
        }
        else {
            return isNull;
        }
    }
    else {
        // Non-nullable types: return constant based on null test type
        // LingoDB pattern: IS_NOT_NULL returns true, IS_NULL returns false for non-nullable
        return ctx.builder.create<mlir::db::ConstantOp>(
            ctx.builder.getUnknownLoc(),
            ctx.builder.getI1Type(),
            ctx.builder.getIntegerAttr(ctx.builder.getI1Type(), null_test->nulltesttype == PG_IS_NOT_NULL));
    }
}

auto PostgreSQLASTTranslator::Impl::translate_aggref(const QueryCtxT& ctx, const Aggref* aggref) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!aggref) {
        PGX_ERROR("Invalid Aggref parameters");
        return nullptr;
    }

    // Aggregate functions are handled differently - they need to be in aggregation context
    PGX_WARNING("Aggref translation requires aggregation context");

    return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                          DEFAULT_PLACEHOLDER_INT,
                                                          ctx.builder.getI64Type());
}

auto PostgreSQLASTTranslator::Impl::translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!coalesce_expr) {
        PGX_ERROR("Invalid CoalesceExpr parameters");
        return nullptr;
    }

    // COALESCE returns first non-null argument
    if (!coalesce_expr->args || coalesce_expr->args->length == 0) {
        PGX_WARNING("COALESCE with no arguments");
        // No arguments - return NULL with default type
        auto nullType = mlir::db::NullableType::get(&context_, ctx.builder.getI32Type());
        return ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), nullType);
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE has %d arguments", coalesce_expr->args->length);

    // First, translate all arguments
    std::vector<mlir::Value> translatedArgs;

    ListCell* cell;
    foreach (cell, coalesce_expr->args) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating COALESCE argument");
        const auto expr = static_cast<Expr*>(lfirst(cell));
        if (mlir::Value val = translate_expression(ctx, expr)) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Argument translated successfully");
            translatedArgs.push_back(val);
        }
        else {
            PGX_WARNING("Failed to translate COALESCE argument");
        }
    }

    if (translatedArgs.empty()) {
        PGX_WARNING("All COALESCE arguments failed to translate");
        auto nullType = mlir::db::NullableType::get(&context_, ctx.builder.getI32Type());
        return ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), nullType);
    }

    // Determine common type following LingoDB pattern
    // Only create nullable result if at least one argument is nullable
    mlir::Type baseType = nullptr;

    for (const auto& arg : translatedArgs) {
        auto argType = arg.getType();
        if (auto nullableType = dyn_cast<mlir::db::NullableType>(argType)) {
            if (!baseType) {
                baseType = nullableType.getType();
            }
        }
        else {
            if (!baseType) {
                baseType = argType;
            }
        }
    }

    // COALESCE should always produce nullable type in query contexts
    // Even when all inputs are non-nullable, the result needs nullable wrapper
    // for proper MaterializeOp handling
    mlir::Type commonType = mlir::db::NullableType::get(&context_, baseType);

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE common type determined - forcing nullable for query context");

    // Now convert arguments to common type only if necessary
    for (size_t i = 0; i < translatedArgs.size(); ++i) {
        if (auto& val = translatedArgs[i]; val.getType() != commonType) {
            // Need to convert to common type
            if (!isa<mlir::db::NullableType>(val.getType())) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "Wrapping non-nullable argument %zu to match common nullable type", i);
                // Wrap non-nullable value in nullable type with explicit false null flag
                auto falseFlag = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(), 0, 1);
                val = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(),
                                                                 commonType, // Result type (nullable)
                                                                 val, // Value to wrap
                                                                 falseFlag // Explicit null flag = false (NOT NULL)
                );
                translatedArgs[i] = val;
            }
        }
    }

    // COALESCE using simplified recursive pattern that's safer
    std::function<mlir::Value(size_t)> buildCoalesceRecursive = [&](const size_t index) -> mlir::Value {
        const auto loc = ctx.builder.getUnknownLoc();

        // Base case: if we're at the last argument, return it
        if (index >= translatedArgs.size() - 1) {
            return translatedArgs.back();
        }

        // Get current argument
        mlir::Value value = translatedArgs[index];

        // Create null check - follow LingoDB semantics exactly
        mlir::Value isNull = ctx.builder.create<mlir::db::IsNullOp>(loc, value);
        mlir::Value isNotNull = ctx.builder.create<mlir::db::NotOp>(loc, isNull);

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

    // Log result type info
    const bool resultIsNullable = mlir::isa<mlir::db::NullableType>(result.getType());
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE final result is nullable: %d", resultIsNullable);

    // COALESCE always returns nullable type for query context compatibility
    // No unpacking needed - MaterializeOp requires nullable types
    const bool resultIsNullableType = isa<mlir::db::NullableType>(result.getType());
    PGX_LOG(AST_TRANSLATE, IO, "translate_coalesce_expr OUT: MLIR Value (nullable=%d)", resultIsNullableType);

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_scalar_array_op_expr(const QueryCtxT& ctx,
                                                                   const ScalarArrayOpExpr* scalar_array_op)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!scalar_array_op) {
        PGX_ERROR("Invalid ScalarArrayOpExpr parameters");
        return nullptr;
    }

    const List* args = scalar_array_op->args;
    if (!args || args->length != 2) {
        PGX_ERROR("ScalarArrayOpExpr: Expected 2 arguments, got %d", args ? args->length : 0);
        return nullptr;
    }

    const auto leftNode = static_cast<Node*>(lfirst(&args->elements[0]));
    mlir::Value leftValue = translate_expression(ctx, reinterpret_cast<Expr*>(leftNode));
    if (!leftValue) {
        PGX_ERROR("Failed to translate left operand of IN expression");
        return nullptr;
    }

    const auto rightNode = static_cast<Node*>(lfirst(&args->elements[1]));

    PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr: Right operand nodeTag = %d", nodeTag(rightNode));

    // Extract array elements into a common format
    std::vector<mlir::Value> arrayElements;

    if (nodeTag(rightNode) == T_ArrayExpr) {
        const auto arrayExpr = reinterpret_cast<ArrayExpr*>(rightNode);
        const List* elements = arrayExpr->elements;

        if (elements) {
            for (int i = 0; i < elements->length; i++) {
                const auto elemNode = static_cast<Node*>(lfirst(&elements->elements[i]));
                if (mlir::Value elemValue = translate_expression(ctx, reinterpret_cast<Expr*>(elemNode))) {
                    arrayElements.push_back(elemValue);
                }
            }
        }
    }
    else if (nodeTag(rightNode) == T_Const) {
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
                                                                                    intValue,
                                                                                    ctx.builder.getI32Type());
                    arrayElements.push_back(elemValue);
                }
            }
        }
        else if (constNode->consttype == PG_TEXT_ARRAY_OID) {
            // Handle text array for IN ('string1', 'string2', ...)
            const auto array = DatumGetArrayTypeP(constNode->constvalue);
            int nitems;
            Datum* values;
            bool* nulls;

            deconstruct_array(array, TEXTOID, -1, false, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    const auto textValue = DatumGetTextP(values[i]);
                    std::string str_value(VARDATA(textValue), VARSIZE(textValue) - VARHDRSZ);

                    auto elemValue = ctx.builder.create<mlir::db::ConstantOp>(ctx.builder.getUnknownLoc(),
                                                                              ctx.builder.getType<mlir::db::StringType>(),
                                                                              ctx.builder.getStringAttr(str_value));
                    arrayElements.push_back(elemValue);
                }
            }
        }
        else {
            PGX_WARNING("ScalarArrayOpExpr: Unsupported const array type %u", constNode->consttype);
        }
    }
    else {
        PGX_WARNING("ScalarArrayOpExpr: Unexpected right operand type %d", nodeTag(rightNode));
    }

    // Handle empty array
    if (arrayElements.empty()) {
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                              scalar_array_op->useOr ? 0 : 1,
                                                              ctx.builder.getI1Type());
    }

    // Build comparison chain
    mlir::Value result = nullptr;

    for (auto elemValue : arrayElements) {
        mlir::Value cmp = nullptr;

        if (scalar_array_op->opno == PG_INT4_EQ_OID || scalar_array_op->opno == PG_INT8_EQ_OID
            || scalar_array_op->opno == PG_INT2_EQ_OID || scalar_array_op->opno == PG_TEXT_EQ_OID)
        {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(),
                                                      mlir::db::DBCmpPredicate::eq,
                                                      leftValue,
                                                      elemValue);
        }
        else if (scalar_array_op->opno == PG_INT4_NE_OID || scalar_array_op->opno == PG_INT8_NE_OID
                 || scalar_array_op->opno == PG_INT2_NE_OID || scalar_array_op->opno == PG_TEXT_NE_OID)
        {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(),
                                                      mlir::db::DBCmpPredicate::neq,
                                                      leftValue,
                                                      elemValue);
        }
        else {
            PGX_WARNING("Unsupported operator OID %u in IN expression, defaulting to equality", scalar_array_op->opno);
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(),
                                                      mlir::db::DBCmpPredicate::eq,
                                                      leftValue,
                                                      elemValue);
        }

        if (!cmp.getType().isInteger(1)) {
            cmp = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), cmp);
        }

        if (!result) {
            result = cmp;
        }
        else {
            if (scalar_array_op->useOr) {
                result = ctx.builder.create<mlir::db::OrOp>(ctx.builder.getUnknownLoc(),
                                                            ctx.builder.getI1Type(),
                                                            mlir::ValueRange{result, cmp});
            }
            else {
                result = ctx.builder.create<mlir::db::AndOp>(ctx.builder.getUnknownLoc(),
                                                             ctx.builder.getI1Type(),
                                                             mlir::ValueRange{result, cmp});
            }
        }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translate_scalar_array_op_expr OUT: MLIR Value");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!case_expr) {
        PGX_ERROR("Invalid CaseExpr parameters");
        return nullptr;
    }

    // CASE expressions in PostgreSQL come in two forms:
    // 1. Simple: CASE expr WHEN val1 THEN result1 WHEN val2 THEN result2 ELSE default END
    // 2. Searched: CASE WHEN cond1 THEN result1 WHEN cond2 THEN result2 ELSE default END

    // Check if this is a simple CASE (has an arg) or searched CASE (no arg)
    mlir::Value caseArg = nullptr;
    if (case_expr->arg) {
        caseArg = translate_expression(ctx, case_expr->arg);
        if (!caseArg) {
            PGX_ERROR("Failed to translate CASE argument expression");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Simple CASE expression with comparison argument");
    }
    else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Searched CASE expression (no comparison argument)");
    }

    // Build nested if-then-else structure from WHEN clauses
    // We'll build this bottom-up, starting with the ELSE clause
    mlir::Value elseResult = nullptr;
    if (case_expr->defresult) {
        elseResult = translate_expression(ctx, case_expr->defresult);
        if (!elseResult) {
            PGX_ERROR("Failed to translate CASE ELSE expression");
            return nullptr;
        }
    }
    else {
        // If no ELSE clause, use NULL as default
        // Create a nullable i32 type for the NULL result
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
                continue;
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
                    continue;
                }
                condition = whenCondition;
            }
            else {
                // Searched CASE: whenClause->expr is the condition itself
                condition = translate_expression(ctx, whenClause->expr);
                if (!condition) {
                    PGX_ERROR("Failed to translate WHEN condition");
                    continue;
                }
            }

            // Ensure condition is boolean
            if (auto conditionType = condition.getType();
                !isa<mlir::IntegerType>(conditionType) || cast<mlir::IntegerType>(conditionType).getWidth() != 1)
            {
                // Need to convert to boolean using db.derive_truth
                condition = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), condition);
            }

            // Translate the THEN result
            mlir::Value thenResult = translate_expression(ctx, whenClause->result);
            if (!thenResult) {
                PGX_ERROR("Failed to translate THEN result");
                continue;
            }

            // Ensure both branches return the same type
            // If types don't match, we need to ensure they're compatible
            auto resultType = result.getType();

            // If one is nullable and the other isn't, make both nullable
            if (auto thenType = thenResult.getType(); resultType != thenType) {
                // Check if one is nullable and the other isn't
                const bool resultIsNullable = isa<mlir::db::NullableType>(resultType);

                if (const bool thenIsNullable = isa<mlir::db::NullableType>(thenType); resultIsNullable && !thenIsNullable)
                {
                    // Wrap thenResult in nullable
                    auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), thenType);
                    thenResult =
                        ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableType, thenResult);
                }
                else if (!resultIsNullable && thenIsNullable) {
                    // Wrap result in nullable
                    auto nullableType = mlir::db::NullableType::get(ctx.builder.getContext(), resultType);
                    result =
                        ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableType, result);
                    resultType = nullableType;
                }
            }

            // Create if-then-else for this WHEN clause
            auto ifOp = ctx.builder.create<mlir::scf::IfOp>(ctx.builder.getUnknownLoc(),
                                                            thenResult.getType(),
                                                            condition,
                                                            true); // Has else region

            // Build THEN region
            ctx.builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), thenResult);

            // Build ELSE region (contains the previous result)
            ctx.builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), result);

            // Move insertion point after if operation
            ctx.builder.setInsertionPointAfter(ifOp);

            // If operation's result becomes our new result
            result = ifOp.getResult(0);
        }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translate_case_expr OUT: MLIR Value (CASE expression)");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_expression_with_case_test(const QueryCtxT& ctx,
                                                                        Expr* expr,
                                                                        const mlir::Value case_test_value)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!expr) {
        return nullptr;
    }

    // If this is a CaseTestExpr, return the case test value
    if (expr->type == T_CaseTestExpr) {
        return case_test_value;
    }

    // For other expression types, we need to recursively replace CaseTestExpr
    // For now, handle the most common case: direct comparison expressions
    if (expr->type == T_OpExpr) {
        const auto opExpr = reinterpret_cast<OpExpr*>(expr);

        // Translate the operation, but replace any CaseTestExpr with the case test value
        if (!opExpr->args || opExpr->args->length != 2) {
            PGX_ERROR("OpExpr in CASE requires exactly 2 arguments");
            return nullptr;
        }

        const auto leftNode = static_cast<Node*>(lfirst(&opExpr->args->elements[0]));
        const auto rightNode = static_cast<Node*>(lfirst(&opExpr->args->elements[1]));

        const mlir::Value leftValue = (leftNode && leftNode->type == T_CaseTestExpr)
                                          ? case_test_value
                                          : translate_expression(ctx, reinterpret_cast<Expr*>(leftNode));
        const mlir::Value rightValue = (rightNode && rightNode->type == T_CaseTestExpr)
                                           ? case_test_value
                                           : translate_expression(ctx, reinterpret_cast<Expr*>(rightNode));

        if (!leftValue || !rightValue) {
            PGX_ERROR("Failed to translate operands in CASE OpExpr");
            return nullptr;
        }

        // Create the comparison operation
        return translate_comparison_op(ctx, opExpr->opno, leftValue, rightValue);
    }

    // For other types, just translate normally (no CaseTestExpr replacement needed)
    return translate_expression(ctx, expr);
}

auto PostgreSQLASTTranslator::Impl::extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr)
    -> std::optional<std::pair<mlir::Value, mlir::Value>> {
    PGX_IO(AST_TRANSLATE);
    if (!op_expr || !op_expr->args) {
        PGX_ERROR("OpExpr has no arguments");
        return std::nullopt;
    }

    if (op_expr->args->length < 1) {
        return std::nullopt;
    }

    // Safety check for elements array (PostgreSQL 17)
    if (!op_expr->args->elements) {
        PGX_WARNING("OpExpr args list has length %d but no elements array", op_expr->args->length);
        PGX_WARNING("This suggests the test setup needs to properly initialize the List structure");
        return std::nullopt;
    }

    mlir::Value lhs;
    mlir::Value rhs;

    // Iterate using PostgreSQL 17 style with elements array
    for (int argIndex = 0; argIndex < op_expr->args->length && argIndex < MAX_BINARY_OPERANDS; argIndex++) {
        const ListCell* lc = &op_expr->args->elements[argIndex];
        if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
            if (const mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
                if (argIndex == LEFT_OPERAND_INDEX) {
                    lhs = argValue;
                }
                else if (argIndex == RIGHT_OPERAND_INDEX) {
                    rhs = argValue;
                }
            }
        }
    }

    // If we couldn't extract proper operands, create placeholders
    if (!lhs) {
        PGX_WARNING("Failed to translate left operand, using placeholder");
        lhs = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                             DEFAULT_PLACEHOLDER_INT,
                                                             ctx.builder.getI32Type());
    }
    if (!rhs && op_expr->args->length >= MAX_BINARY_OPERANDS) {
        PGX_WARNING("Failed to translate right operand, using placeholder");
        rhs = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                             DEFAULT_PLACEHOLDER_INT,
                                                             ctx.builder.getI32Type());
    }

    if (!lhs || !rhs) {
        return std::nullopt;
    }

    return std::make_pair(lhs, rhs);
}

auto PostgreSQLASTTranslator::Impl::translate_arithmetic_op(const QueryCtxT& ctx,
                                                            const Oid op_oid,
                                                            const mlir::Value lhs,
                                                            const mlir::Value rhs) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    switch (op_oid) {
    // Addition operators
    case PG_INT4_PLUS_OID:
    case PG_INT8_PLUS_OID: return ctx.builder.create<mlir::db::AddOp>(ctx.builder.getUnknownLoc(), lhs, rhs);

    case PG_INT4_MINUS_OID:
    case PG_INT4_MINUS_ALT_OID:
    case PG_INT8_MINUS_OID: return ctx.builder.create<mlir::db::SubOp>(ctx.builder.getUnknownLoc(), lhs, rhs);

    case PG_INT4_MUL_OID:
    case PG_INT8_MUL_OID: return ctx.builder.create<mlir::db::MulOp>(ctx.builder.getUnknownLoc(), lhs, rhs);

    case PG_INT4_DIV_OID:
    case PG_INT4_DIV_ALT_OID:
    case PG_INT8_DIV_OID: return ctx.builder.create<mlir::db::DivOp>(ctx.builder.getUnknownLoc(), lhs, rhs);

    case PG_INT4_MOD_OID:
    case PG_INT4_MOD_ALT_OID:
    case PG_INT8_MOD_OID: return ctx.builder.create<mlir::db::ModOp>(ctx.builder.getUnknownLoc(), lhs, rhs);

    default: return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translate_comparison_op(const QueryCtxT& ctx,
                                                            const Oid op_oid,
                                                            const mlir::Value lhs,
                                                            const mlir::Value rhs) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    mlir::db::DBCmpPredicate predicate;

    switch (op_oid) {
    case PG_INT4_EQ_OID:
    case PG_INT8_EQ_OID:
    case PG_TEXT_EQ_OID: predicate = mlir::db::DBCmpPredicate::eq; break;

    case PG_INT4_NE_OID:
    case PG_INT8_NE_OID:
    case PG_TEXT_NE_OID: predicate = mlir::db::DBCmpPredicate::neq; break;

    case PG_INT4_LT_OID:
    case PG_INT8_LT_OID:
    case PG_TEXT_LT_OID: predicate = mlir::db::DBCmpPredicate::lt; break;

    case PG_INT4_LE_OID:
    case PG_INT8_LE_OID:
    case PG_TEXT_LE_OID: predicate = mlir::db::DBCmpPredicate::lte; break;

    case PG_INT4_GT_OID:
    case PG_INT8_GT_OID:
    case PG_TEXT_GT_OID: predicate = mlir::db::DBCmpPredicate::gt; break;

    case PG_INT4_GE_OID:
    case PG_INT8_GE_OID:
    case PG_TEXT_GE_OID: predicate = mlir::db::DBCmpPredicate::gte; break;

    default: return nullptr;
    }

    mlir::Value convertedLhs = lhs;
    mlir::Value convertedRhs = rhs;

    const bool lhsNullable = isa<mlir::db::NullableType>(lhs.getType());
    const bool rhsNullable = isa<mlir::db::NullableType>(rhs.getType());

    if (lhsNullable && !rhsNullable) {
        mlir::Type nullableRhsType = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
        auto falseVal = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(), 0, 1);
        convertedRhs =
            ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableRhsType, rhs, falseVal);
    }
    else if (!lhsNullable && rhsNullable) {
        mlir::Type nullableLhsType = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
        auto falseVal = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(), 0, 1);
        convertedLhs =
            ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullableLhsType, lhs, falseVal);
    }

    return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}

} // namespace postgresql_ast