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

auto PostgreSQLASTTranslator::Impl::extract_op_expr_operands(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                             const OptRefT<const TranslationResult> current_result)
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
        if (const auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return nullable.getType();
        }
        return t;
    };

    const bool lhs_is_string = mlir::isa<mlir::db::StringType>(get_base_type(lhs.getType()));
    const bool rhs_is_string = mlir::isa<mlir::db::StringType>(get_base_type(rhs.getType()));

    if (!lhs_is_string || !rhs_is_string) {
        return {lhs, rhs};
    }

    auto* lhs_expr = static_cast<Expr*>(lfirst(&op_expr->args->elements[0]));
    auto* rhs_expr = static_cast<Expr*>(lfirst(&op_expr->args->elements[1]));

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

    auto pad_string_constant = [&](const mlir::Value val, const int target_length) -> mlir::Value {
        auto* defOp = val.getDefiningOp();
        if (!defOp || !mlir::isa<mlir::db::ConstantOp>(defOp)) {
            return val;
        }

        auto constOp = mlir::cast<mlir::db::ConstantOp>(defOp);
        if (const auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constOp.getValue())) {
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

    if (lhs_bpchar_len > 0 && nodeTag(rhs_expr) == T_Const) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Normalizing RHS constant to match LHS BPCHAR(%d)", lhs_bpchar_len);
        rhs = pad_string_constant(rhs, lhs_bpchar_len);
    } else if (rhs_bpchar_len > 0 && nodeTag(lhs_expr) == T_Const) {
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
        if (const auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
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
        const bool isNullable = isa<mlir::db::NullableType>(v.getType());
        if (isNullable && !isa<mlir::db::NullableType>(t)) {
            t = mlir::db::NullableType::get(builder.getContext(), t);
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
        } else {
            return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
        }
    }
    static mlir::Type getCommonBaseType(mlir::Type left, mlir::Type right) {
        left = getBaseType(left);
        right = getBaseType(right);

        const bool leftIsDate = isa<mlir::db::DateType>(left);
        const bool rightIsDate = isa<mlir::db::DateType>(right);
        const bool leftIsTimestamp = isa<mlir::db::TimestampType>(left);
        const bool rightIsTimestamp = isa<mlir::db::TimestampType>(right);

        if ((leftIsDate || leftIsTimestamp) && (rightIsDate || rightIsTimestamp)) {
            if (leftIsTimestamp)
                return left;
            if (rightIsTimestamp)
                return right;
            return left;
        }

        const bool stringPresent = isa<mlir::db::StringType>(left) || isa<mlir::db::StringType>(right);
        const bool intPresent = isa<mlir::IntegerType>(left) || isa<mlir::IntegerType>(right);
        const bool floatPresent = isa<mlir::FloatType>(left) || isa<mlir::FloatType>(right);
        const bool decimalPresent = isa<mlir::db::DecimalType>(left) || isa<mlir::db::DecimalType>(right);
        if (stringPresent)
            return mlir::db::StringType::get(left.getContext());
        if (decimalPresent)
            return getHigherDecimalType(left, right);
        if (floatPresent)
            return static_cast<mlir::Type>(getHigherFloatType(left, right));
        if (intPresent)
            return getHigherIntType(left, right);
        return left;
    }
    static mlir::Type getCommonType(const mlir::Type left, const mlir::Type right) {
        const bool isNullable = isa<mlir::db::NullableType>(left) || isa<mlir::db::NullableType>(right);
        const auto commonBaseType = getCommonBaseType(left, right);
        if (isNullable) {
            return mlir::db::NullableType::get(left.getContext(), commonBaseType);
        } else {
            return commonBaseType;
        }
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

auto PostgreSQLASTTranslator::Impl::verify_and_print(const mlir::Value val) -> void {
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
        std::string valueStr;
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
}

auto PostgreSQLASTTranslator::Impl::print_type(const mlir::Type val) -> void {
    std::string valueStr;
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
    return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}
} // namespace postgresql_ast