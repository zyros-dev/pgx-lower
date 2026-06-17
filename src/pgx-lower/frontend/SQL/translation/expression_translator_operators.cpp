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
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <string>

namespace mlir::relalg {
class GetColumnOp;
} // namespace mlir::relalg

namespace postgresql_ast {
using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLASTTranslator::Impl::translate_op_expr(const QueryCtxT& ctx, const OpExpr* op_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!op_expr) {
        PGX_ERROR("Invalid OpExpr parameters");
        throw std::runtime_error("Invalid OpExpr parameters");
    }

    auto operands = extract_op_expr_operands(ctx, op_expr);
    if (!operands) {
        PGX_ERROR("Failed to extract OpExpr operands");
        throw std::runtime_error("Invalid OpExpr parameters");
    }

    auto [lhs, rhs] = *operands;
    const Oid opOid = op_expr->opno;

    {
        if (auto result = translate_arithmetic_op(ctx, op_expr, lhs, rhs)) {
            return result;
        }
    }

    {
        if (auto result = translate_comparison_op(ctx, opOid, lhs, rhs)) {
            return result;
        }
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

            llvm::SmallVector<mlir::Value, 2> operands{convertedLhs, convertedRhs};
            auto resultType = pgx_lower::frontend::sql::sql_bool_result_type(ctx.builder, operands);

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                 ctx.builder.getStringAttr("Like"), operands);

            return op2.getRes();
        }
        if (op == "!~~") {
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

            llvm::SmallVector<mlir::Value, 2> operands{convertedLhs, convertedRhs};
            auto resultType = pgx_lower::frontend::sql::sql_bool_result_type(ctx.builder, operands);

            auto likeOp = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), resultType,
                                                                    ctx.builder.getStringAttr("Like"), operands);

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

auto PostgreSQLASTTranslator::Impl::extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr)
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

    for (int argIndex{}; argIndex < op_expr->args->length && argIndex < 2; argIndex++) {
        const ListCell* lc = &op_expr->args->elements[argIndex];
        if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
            if (const mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
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
    auto get_base_type = [](mlir::Type t) -> mlir::Type {
        if (const auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return nullable.getType();
        }
        return t;
    };

    const bool has_date_or_interval = mlir::isa<mlir::db::DateType, mlir::db::PgDateType>(get_base_type(lhs.getType()))
                                      || mlir::isa<mlir::db::DateType, mlir::db::PgDateType>(get_base_type(rhs.getType()))
                                      || mlir::isa<mlir::db::IntervalType, mlir::db::PgIntervalType>(
                                          get_base_type(lhs.getType()))
                                      || mlir::isa<mlir::db::IntervalType, mlir::db::PgIntervalType>(
                                          get_base_type(rhs.getType()));

    PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC] op=%s, has_date_or_interval=%d, opresulttype=%u", op.c_str(),
            has_date_or_interval, op_expr->opresulttype);

    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, lhs, rhs);

    const bool has_pg_operand = mlir::db::isPgValueType(convertedLhs.getType())
                                || mlir::db::isPgValueType(convertedRhs.getType());
    if (has_date_or_interval || has_pg_operand) {
        const PostgreSQLTypeMapper type_mapper(*ctx.builder.getContext());
        auto result_type = type_mapper.map_postgre_sqltype(op_expr->opresulttype, -1, op_expr->opcollid, false);

        PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC PG] Forcing result type from PostgreSQL opresulttype=%u",
                op_expr->opresulttype);

        const bool lhs_nullable = mlir::isa<mlir::db::NullableType>(convertedLhs.getType())
                                  || (mlir::db::isPgValueType(convertedLhs.getType())
                                      && mlir::db::getPgNullability(convertedLhs.getType())
                                             == mlir::db::PgNullability::Maybe);
        const bool rhs_nullable = mlir::isa<mlir::db::NullableType>(convertedRhs.getType())
                                  || (mlir::db::isPgValueType(convertedRhs.getType())
                                      && mlir::db::getPgNullability(convertedRhs.getType())
                                             == mlir::db::PgNullability::Maybe);
        if (lhs_nullable || rhs_nullable) {
            result_type = mlir::db::isPgValueType(result_type)
                              ? mlir::db::withPgNullability(result_type, mlir::db::PgNullability::Maybe)
                              : mlir::db::NullableType::get(ctx.builder.getContext(), result_type);
        }

        if (op == "+") {
            return ctx.builder.create<mlir::db::AddOp>(loc, result_type, convertedLhs, convertedRhs);
        }
        if (op == "-") {
            return ctx.builder.create<mlir::db::SubOp>(loc, result_type, convertedLhs, convertedRhs);
        }
        if (op == "*") {
            return ctx.builder.create<mlir::db::MulOp>(loc, result_type, convertedLhs, convertedRhs);
        }
        if (op == "/") {
            return ctx.builder.create<mlir::db::DivOp>(loc, result_type, convertedLhs, convertedRhs);
        }
        if (op == "%") {
            return ctx.builder.create<mlir::db::ModOp>(loc, result_type, convertedLhs, convertedRhs);
        }
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
        if (auto rightFloat = dyn_cast_or_null<mlir::FloatType>(right)) {
            if (!leftFloat || rightFloat.getWidth() > leftFloat.getWidth()) {
                return rightFloat;
            }
        }
        return leftFloat;
    }
    static mlir::IntegerType getHigherIntType(mlir::Type left, mlir::Type right) {
        const mlir::IntegerType leftInt = dyn_cast_or_null<mlir::IntegerType>(left);
        if (const auto rightInt = dyn_cast_or_null<mlir::IntegerType>(right)) {
            if (!leftInt || rightInt.getWidth() > leftInt.getWidth()) {
                return rightInt;
            }
        }
        return leftInt;
    }
    static mlir::db::DecimalType getHigherDecimalType(mlir::Type left, mlir::Type right) {
        const auto a = dyn_cast_or_null<mlir::db::DecimalType>(left);
        if (const auto b = dyn_cast_or_null<mlir::db::DecimalType>(right)) {
            if (!a) {
                return b;
            }
            const int hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
            const int maxs = std::max(a.getS(), b.getS());
            return mlir::db::DecimalType::get(a.getContext(), std::min(hidig + maxs, MAX_NUMERIC_PRECISION),
                                              std::min(maxs, MAX_NUMERIC_UNCONSTRAINED_SCALE));
        }
        return a;
    }
    static bool isPgIntegerType(mlir::Type type) {
        return mlir::isa<mlir::db::PgInt2Type, mlir::db::PgInt4Type, mlir::db::PgInt8Type>(type);
    }
    static bool isPgFloatType(mlir::Type type) {
        return mlir::isa<mlir::db::PgFloat4Type, mlir::db::PgFloat8Type>(type);
    }
    static bool isPgStringType(mlir::Type type) {
        return mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType, mlir::db::PgBpcharType>(type);
    }
    static unsigned getIntegerWidth(mlir::Type type) {
        if (const auto intType = dyn_cast_or_null<mlir::IntegerType>(type)) {
            return intType.getWidth();
        }
        if (!isPgIntegerType(type)) {
            return 0;
        }
        return dyn_cast<mlir::IntegerType>(mlir::db::getPgPhysicalCarrierType(type)).getWidth();
    }
    static unsigned getFloatWidth(mlir::Type type) {
        if (auto floatType = dyn_cast_or_null<mlir::FloatType>(type)) {
            return floatType.getWidth();
        }
        if (!isPgFloatType(type)) {
            return 0;
        }
        return dyn_cast<mlir::FloatType>(mlir::db::getPgPhysicalCarrierType(type)).getWidth();
    }
    static int32_t getPgTypmodOrUnconstrained(mlir::Type left, mlir::Type right) {
        if (mlir::db::isPgValueType(left) && mlir::db::getPgTypmod(left) >= 0) {
            return mlir::db::getPgTypmod(left);
        }
        if (mlir::db::isPgValueType(right) && mlir::db::getPgTypmod(right) >= 0) {
            return mlir::db::getPgTypmod(right);
        }
        return -1;
    }
    static mlir::db::PgOid getPgCollationOrInvalid(mlir::Type left, mlir::Type right) {
        if (mlir::db::isPgValueType(left) && mlir::db::getPgCollation(left) != InvalidOid) {
            return mlir::db::getPgCollation(left);
        }
        if (mlir::db::isPgValueType(right) && mlir::db::getPgCollation(right) != InvalidOid) {
            return mlir::db::getPgCollation(right);
        }
        return InvalidOid;
    }
    static mlir::Type getPgIntegerType(mlir::MLIRContext* context, unsigned width) {
        if (width <= 16) {
            return mlir::db::PgInt2Type::get(context);
        }
        if (width <= 32) {
            return mlir::db::PgInt4Type::get(context);
        }
        return mlir::db::PgInt8Type::get(context);
    }
    static mlir::Type getPgFloatType(mlir::MLIRContext* context, unsigned width) {
        if (width <= 32) {
            return mlir::db::PgFloat4Type::get(context);
        }
        return mlir::db::PgFloat8Type::get(context);
    }
    static mlir::Type getCommonPgBaseType(mlir::Type left, mlir::Type right) {
        auto* context = left.getContext();
        if (isPgStringType(left) || isPgStringType(right) || mlir::isa<mlir::db::StringType>(left)
            || mlir::isa<mlir::db::StringType>(right))
        {
            if (mlir::isa<mlir::db::PgBpcharType>(left) || mlir::isa<mlir::db::PgBpcharType>(right)) {
                return mlir::db::PgBpcharType::get(context, getPgTypmodOrUnconstrained(left, right),
                                                   getPgCollationOrInvalid(left, right));
            }
            return mlir::db::PgTextType::get(context, getPgCollationOrInvalid(left, right));
        }
        if (mlir::isa<mlir::db::PgTimestampType, mlir::db::TimestampType>(left)
            || mlir::isa<mlir::db::PgTimestampType, mlir::db::TimestampType>(right))
        {
            return mlir::db::PgTimestampType::get(context, getPgTypmodOrUnconstrained(left, right));
        }
        if (mlir::isa<mlir::db::PgDateType, mlir::db::DateType>(left)
            || mlir::isa<mlir::db::PgDateType, mlir::db::DateType>(right))
        {
            return mlir::db::PgDateType::get(context);
        }
        if (mlir::isa<mlir::db::PgIntervalType, mlir::db::IntervalType>(left)
            || mlir::isa<mlir::db::PgIntervalType, mlir::db::IntervalType>(right))
        {
            return mlir::db::PgIntervalType::get(context, getPgTypmodOrUnconstrained(left, right));
        }
        if (mlir::isa<mlir::db::PgNumericType, mlir::db::DecimalType>(left)
            || mlir::isa<mlir::db::PgNumericType, mlir::db::DecimalType>(right))
        {
            return mlir::db::PgNumericType::get(context, getPgTypmodOrUnconstrained(left, right));
        }
        const unsigned floatWidth = std::max(getFloatWidth(left), getFloatWidth(right));
        if (floatWidth > 0) {
            return getPgFloatType(context, floatWidth);
        }
        const unsigned integerWidth = std::max(getIntegerWidth(left), getIntegerWidth(right));
        if (integerWidth > 0) {
            return getPgIntegerType(context, integerWidth);
        }
        if (mlir::isa<mlir::db::PgBoolType>(left) || mlir::isa<mlir::db::PgBoolType>(right)) {
            return mlir::db::PgBoolType::get(context);
        }
        return mlir::db::isPgValueType(left) ? left : right;
    }
    static mlir::Value castValueToType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
        const bool isNullable = isa<mlir::db::NullableType>(v.getType());
        if (mlir::db::isPgValueType(v.getType()) && mlir::db::isPgValueType(t)) {
            t = mlir::db::withPgNullability(t, mlir::db::getPgNullability(v.getType()));
        } else if (isNullable && !isa<mlir::db::NullableType>(t)) {
            t = mlir::db::isPgValueType(t) ? mlir::db::withPgNullability(t, mlir::db::PgNullability::Maybe)
                                           : mlir::Type(mlir::db::NullableType::get(builder.getContext(), t));
        }
        const bool onlyTargetIsNullable = !isNullable && isa<mlir::db::NullableType>(t);
        if (v.getType() == t) {
            return v;
        }
        if (auto* defOp = v.getDefiningOp()) {
            if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
                if (!isa<mlir::db::NullableType>(t)) {
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
        }
        return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
    }
    static mlir::Type getCommonBaseType(mlir::Type left, mlir::Type right) {
        left = getBaseType(left);
        right = getBaseType(right);

        if (mlir::db::isPgValueType(left) || mlir::db::isPgValueType(right)) {
            return getCommonPgBaseType(left, right);
        }

        const bool leftIsDate = isa<mlir::db::DateType>(left);
        const bool rightIsDate = isa<mlir::db::DateType>(right);
        const bool leftIsTimestamp = isa<mlir::db::TimestampType>(left);
        const bool rightIsTimestamp = isa<mlir::db::TimestampType>(right);

        if ((leftIsDate || leftIsTimestamp) && (rightIsDate || rightIsTimestamp)) {
            if (leftIsTimestamp) {
                return left;
            }
            if (rightIsTimestamp) {
                return right;
            }
            return left;
        }

        const bool stringPresent = isa<mlir::db::StringType>(left) || isa<mlir::db::StringType>(right);
        const bool intPresent = isa<mlir::IntegerType>(left) || isa<mlir::IntegerType>(right);
        const bool floatPresent = isa<mlir::FloatType>(left) || isa<mlir::FloatType>(right);
        const bool decimalPresent = isa<mlir::db::DecimalType>(left) || isa<mlir::db::DecimalType>(right);
        if (stringPresent) {
            return mlir::db::StringType::get(left.getContext());
        }
        if (decimalPresent) {
            return getHigherDecimalType(left, right);
        }
        if (floatPresent) {
            return static_cast<mlir::Type>(getHigherFloatType(left, right));
        }
        if (intPresent) {
            return getHigherIntType(left, right);
        }
        return left;
    }
    static mlir::Type getCommonType(const mlir::Type left, const mlir::Type right) {
        const bool isNullable = isa<mlir::db::NullableType>(left) || isa<mlir::db::NullableType>(right);
        const auto commonBaseType = getCommonBaseType(left, right);
        if (isNullable) {
            if (mlir::db::isPgValueType(commonBaseType)) {
                return mlir::db::withPgNullability(commonBaseType, mlir::db::PgNullability::Maybe);
            }
            return mlir::db::NullableType::get(left.getContext(), commonBaseType);
        }
        return commonBaseType;
    }
    static mlir::Type getCommonBaseType(const mlir::TypeRange types) {
        mlir::Type commonType = types.front();
        for (const auto t : types) {
            commonType = getCommonBaseType(commonType, t);
        }
        return commonType;
    }
    static std::vector<mlir::Value> toCommonBaseTypes(mlir::OpBuilder& builder, const mlir::ValueRange values) {
        const auto commonType = getCommonBaseType(values.getTypes());
        std::vector<mlir::Value> res;
        for (const auto val : values) {
            res.push_back(castValueToType(builder, val, commonType));
        }
        return res;
    }
    static std::vector<mlir::Value>
    toCommonBaseTypesExceptDecimals(mlir::OpBuilder& builder, const mlir::ValueRange values) {
        std::vector<mlir::Value> res;
        for (auto val : values) {
            if (!isa<mlir::db::DecimalType>(getBaseType(val.getType()))) {
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
    auto convertToType = [&ctx](mlir::Value value, mlir::Type targetBaseType, const bool needsNullable) -> mlir::Value {
        const auto currentType = value.getType();
        const auto currentBaseType = getBaseType(currentType);
        const bool isNullable = mlir::isa<mlir::db::NullableType>(currentType);
        const bool isPgValue = mlir::db::isPgValueType(currentType);
        const auto loc = ctx.builder.getUnknownLoc();

        if (currentBaseType != targetBaseType) {
            if (isPgValue && mlir::db::isPgValueType(targetBaseType)) {
                const auto targetType = mlir::db::withPgNullability(targetBaseType,
                                                                    mlir::db::getPgNullability(currentType));
                value = ctx.builder.create<mlir::db::CastOp>(loc, targetType, value);
            } else if (isNullable) {
                const auto targetType = mlir::db::isPgValueType(targetBaseType)
                                            ? mlir::db::withPgNullability(targetBaseType, mlir::db::PgNullability::Maybe)
                                            : mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                                                     targetBaseType));
                value = ctx.builder.create<mlir::db::CastOp>(loc, targetType, value);
            } else {
                value = ctx.builder.create<mlir::db::CastOp>(loc, targetBaseType, value);
            }
        }

        if (needsNullable && !mlir::isa<mlir::db::NullableType>(value.getType())
            && !mlir::db::isPgValueType(value.getType()))
        {
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

auto PostgreSQLASTTranslator::Impl::verify_and_print(const mlir::Value val) -> void {
#ifndef PGX_RELEASE_MODE
    PGX_IO(AST_TRANSLATE);
    if (auto* defOp = val.getDefiningOp()) {
        const auto verifyResult = mlir::verify(defOp);
        if (mlir::failed(verifyResult)) {
            PGX_ERROR("MLIR verification FAILED for value");
            throw std::runtime_error("MLIR verification FAILED for value");
        }
    } else {
        PGX_LOG(AST_TRANSLATE, TRACE, "val had no defining op");
    }

    PGX_LOG(AST_TRANSLATE, TRACE, "finished verification - now printing.");
    try {
        std::string valueStr{};
        llvm::raw_string_ostream stream(valueStr);
        val.print(stream);
        stream.flush();
        if (valueStr.empty()) {
            PGX_LOG(AST_TRANSLATE, TRACE, "<empty print output>");
        } else {
            PGX_LOG(AST_TRANSLATE, TRACE, "%s", valueStr.c_str());
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during value print: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception during value print");
    }
#endif
}

auto PostgreSQLASTTranslator::Impl::print_type(const mlir::Type val) -> void {
    std::string valueStr{};
    llvm::raw_string_ostream stream(valueStr);
    val.print(stream);
    stream.flush();
    PGX_LOG(AST_TRANSLATE, TRACE, "%s", valueStr.c_str());
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
        print_type(lhs.getType());
        print_type(rhs.getType());
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
    if (mlir::db::isPgValueType(convertedLhs.getType()) || mlir::db::isPgValueType(convertedRhs.getType())) {
        auto resultType = mlir::db::PgBoolType::get(
            ctx.builder.getContext(), mlir::db::combineSqlNullability(mlir::ValueRange{convertedLhs, convertedRhs}));
        return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), resultType, predicate, convertedLhs,
                                                   convertedRhs);
    }
    return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}
} // namespace postgresql_ast
