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
    std::tie(lhs, rhs) = normalize_bpchar_operands(ctx, op_expr, lhs, rhs);
    const Oid OP_OID = op_expr->opno;

    {
        if (auto result = translate_arithmetic_op(ctx, op_expr, lhs, rhs)) {
            return result;
}
    }

    {
        if (auto result = translate_comparison_op(ctx, OP_OID, lhs, rhs)) {
            return result;
}
    }

    if (auto* oprname = get_opname(OP_OID)) {
        std::string const op(oprname);
        pfree(oprname);

        if (op == "~~") {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LIKE operator to db.runtime_call");

            auto converted_lhs = lhs;
            auto converted_rhs = rhs;

            const auto LHS_NULLABLE = isa<mlir::db::NullableType>(lhs.getType());
            const auto RHS_NULLABLE = isa<mlir::db::NullableType>(rhs.getType());

            if (LHS_NULLABLE && !RHS_NULLABLE) {
                auto nullable_rhs_type = mlir::db::NullableType::get(ctx.builder.getContext(), rhs.getType());
                converted_rhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullable_rhs_type,
                                                                          rhs);
            } else if (!LHS_NULLABLE && RHS_NULLABLE) {
                auto nullable_lhs_type = mlir::db::NullableType::get(ctx.builder.getContext(), lhs.getType());
                converted_lhs = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullable_lhs_type,
                                                                          lhs);
            }

            const bool HAS_NULLABLE_OPERAND = LHS_NULLABLE || RHS_NULLABLE;
            auto result_type = HAS_NULLABLE_OPERAND ? mlir::Type(mlir::db::NullableType::get(ctx.builder.getContext(),
                                                                                          ctx.builder.getI1Type()))
                                                 : mlir::Type(ctx.builder.getI1Type());

            auto op2 = ctx.builder.create<mlir::db::RuntimeCall>(ctx.builder.getUnknownLoc(), result_type,
                                                                 ctx.builder.getStringAttr("Like"),
                                                                 mlir::ValueRange{converted_lhs, converted_rhs});

            return op2.getRes();
        } if (op == "!~~") {
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

    PGX_ERROR("Unsupported operator OID: %d", OP_OID);
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

    for (int arg_index = 0; arg_index < op_expr->args->length && arg_index < 2; arg_index++) {
        const ListCell* const lc = &op_expr->args->elements[arg_index];
        if (auto *const ARG_NODE = static_cast<Node*>(lfirst(lc))) {
            if (const mlir::Value ARG_VALUE = translate_expression(ctx, reinterpret_cast<Expr*>(ARG_NODE))) {
                if (arg_index == 0) {
                    lhs = ARG_VALUE;
                } else if (arg_index == 1) {
                    rhs = ARG_VALUE;
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
        if (const auto NULLABLE = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return NULLABLE.getType();
        }
        return t;
    };

    const bool LHS_IS_STRING = mlir::isa<mlir::db::StringType>(get_base_type(lhs.getType()));
    const bool RHS_IS_STRING = mlir::isa<mlir::db::StringType>(get_base_type(rhs.getType()));

    if (!LHS_IS_STRING || !RHS_IS_STRING) {
        return {lhs, rhs};
    }

    auto* lhs_expr = static_cast<Expr*>(lfirst(&op_expr->args->elements[0]));
    auto* rhs_expr = static_cast<Expr*>(lfirst(&op_expr->args->elements[1]));

    auto extract_bpchar_length = [](Expr* expr) -> int {
        if (!expr) {
            return -1;
        }
        const Oid TYPE_OID = exprType(reinterpret_cast<Node*>(expr));
        const int32 TYPE_MOD = exprTypmod(reinterpret_cast<Node*>(expr));

        if (TYPE_OID == BPCHAROID && TYPE_MOD >= VARHDRSZ) {
            return TYPE_MOD - VARHDRSZ;
        }
        return -1;
    };

    auto pad_string_constant = [&](const mlir::Value VAL, const int TARGET_LENGTH) -> mlir::Value {
        auto* def_op = VAL.getDefiningOp();
        if (!def_op || !mlir::isa<mlir::db::ConstantOp>(def_op)) {
            return VAL;
        }

        auto const_op = mlir::cast<mlir::db::ConstantOp>(def_op);
        if (const auto STR_ATTR = mlir::dyn_cast<mlir::StringAttr>(const_op.getValue())) {
            std::string str_value = STR_ATTR.getValue().str();
            if (static_cast<int>(str_value.length()) < TARGET_LENGTH) {
                str_value.resize(TARGET_LENGTH, ' ');
                PGX_LOG(AST_TRANSLATE, DEBUG, "Padded BPCHAR constant to length %d: '%s'", TARGET_LENGTH,
                        str_value.c_str());

                return ctx.builder.create<mlir::db::ConstantOp>(ctx.builder.getUnknownLoc(),
                                                                ctx.builder.getType<mlir::db::StringType>(),
                                                                ctx.builder.getStringAttr(str_value));
            }
        }
        return VAL;
    };

    const int LHS_BPCHAR_LEN = extract_bpchar_length(lhs_expr);
    const int RHS_BPCHAR_LEN = extract_bpchar_length(rhs_expr);

    if (LHS_BPCHAR_LEN > 0 && nodeTag(rhs_expr) == T_Const) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Normalizing RHS constant to match LHS BPCHAR(%d)", LHS_BPCHAR_LEN);
        rhs = pad_string_constant(rhs, LHS_BPCHAR_LEN);
    } else if (RHS_BPCHAR_LEN > 0 && nodeTag(lhs_expr) == T_Const) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Normalizing LHS constant to match RHS BPCHAR(%d)", RHS_BPCHAR_LEN);
        lhs = pad_string_constant(lhs, RHS_BPCHAR_LEN);
    }

    return {lhs, rhs};
}

auto PostgreSQLASTTranslator::Impl::translate_arithmetic_op(const QueryCtxT& ctx, const OpExpr* op_expr,
                                                            const mlir::Value LHS, const mlir::Value RHS) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!op_expr) {
        PGX_ERROR("Invalid OpExpr");
        throw std::runtime_error("Invalid OpExpr");
    }

    const Oid OP_OID = op_expr->opno;
    char* const oprname = get_opname(OP_OID);
    if (!oprname) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unknown arithmetic operator OID: %d", OP_OID);
        throw std::runtime_error("Check logs");
    }

    const std::string OP(oprname);
    pfree(oprname);

    if (OP != "+" && OP != "-" && OP != "*" && OP != "/" && OP != "%") {
        return nullptr;
    }

    const auto LOC = ctx.builder.getUnknownLoc();

    // Check if we need to override the result type (for date/interval arithmetic)
    auto get_base_type = [](mlir::Type t) -> mlir::Type {
        if (const auto NULLABLE = mlir::dyn_cast<mlir::db::NullableType>(t)) {
            return NULLABLE.getType();
        }
        return t;
    };

    const bool HAS_DATE_OR_INTERVAL = mlir::isa<mlir::db::DateType>(get_base_type(LHS.getType()))
                                      || mlir::isa<mlir::db::DateType>(get_base_type(RHS.getType()))
                                      || mlir::isa<mlir::db::IntervalType>(get_base_type(LHS.getType()))
                                      || mlir::isa<mlir::db::IntervalType>(get_base_type(RHS.getType()));

    PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC] op=%s, has_date_or_interval=%d, opresulttype=%u", OP.c_str(),
            HAS_DATE_OR_INTERVAL, op_expr->opresulttype);

    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, LHS, RHS);

    if (HAS_DATE_OR_INTERVAL) {
        const PostgreSQLTypeMapper TYPE_MAPPER(*ctx.builder.getContext());
        auto result_type = TYPE_MAPPER.map_postgre_sqltype(op_expr->opresulttype, -1, false);

        PGX_LOG(AST_TRANSLATE, DEBUG, "[ARITHMETIC DATE] Forcing result type from PostgreSQL opresulttype=%u",
                op_expr->opresulttype);

        const bool LHS_NULLABLE = mlir::isa<mlir::db::NullableType>(convertedLhs.getType());
        const bool RHS_NULLABLE = mlir::isa<mlir::db::NullableType>(convertedRhs.getType());
        if (LHS_NULLABLE || RHS_NULLABLE) {
            result_type = mlir::db::NullableType::get(ctx.builder.getContext(), result_type);
        }

        if (OP == "+") {
            return ctx.builder.create<mlir::db::AddOp>(LOC, result_type, convertedLhs, convertedRhs);
}
        if (OP == "-") {
            return ctx.builder.create<mlir::db::SubOp>(LOC, result_type, convertedLhs, convertedRhs);
}
        if (OP == "*") {
            return ctx.builder.create<mlir::db::MulOp>(LOC, result_type, convertedLhs, convertedRhs);
}
        if (OP == "/") {
            return ctx.builder.create<mlir::db::DivOp>(LOC, result_type, convertedLhs, convertedRhs);
}
        if (OP == "%") {
            return ctx.builder.create<mlir::db::ModOp>(LOC, result_type, convertedLhs, convertedRhs);
}
    } else {
        mlir::SmallVector<mlir::Type, 1> inferred_types;

        if (OP == "+") {
            if (mlir::failed(mlir::db::AddOp::inferReturnTypes(
                    ctx.builder.getContext(), LOC, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferred_types)))
            {
                PGX_ERROR("Failed to infer AddOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::AddOp>(LOC, inferred_types[0], convertedLhs, convertedRhs);
        }
        if (OP == "-") {
            if (mlir::failed(mlir::db::SubOp::inferReturnTypes(
                    ctx.builder.getContext(), LOC, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferred_types)))
            {
                PGX_ERROR("Failed to infer SubOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::SubOp>(LOC, inferred_types[0], convertedLhs, convertedRhs);
        }
        if (OP == "*") {
            if (mlir::failed(mlir::db::MulOp::inferReturnTypes(
                    ctx.builder.getContext(), LOC, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferred_types)))
            {
                PGX_ERROR("Failed to infer MulOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::MulOp>(LOC, inferred_types[0], convertedLhs, convertedRhs);
        }
        if (OP == "/") {
            if (mlir::failed(mlir::db::DivOp::inferReturnTypes(
                    ctx.builder.getContext(), LOC, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferred_types)))
            {
                PGX_ERROR("Failed to infer DivOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::DivOp>(LOC, inferred_types[0], convertedLhs, convertedRhs);
        }
        if (OP == "%") {
            if (mlir::failed(mlir::db::ModOp::inferReturnTypes(
                    ctx.builder.getContext(), LOC, {convertedLhs, convertedRhs}, nullptr, nullptr, {}, inferred_types)))
            {
                PGX_ERROR("Failed to infer ModOp return type");
                throw std::runtime_error("Check logs");
            }
            return ctx.builder.create<mlir::db::ModOp>(LOC, inferred_types[0], convertedLhs, convertedRhs);
        }
    }

    PGX_ERROR("Failed to create arithmetic operation for operator: %s (OID: %d)", OP.c_str(), OP_OID);
    throw std::runtime_error("Check logs");
}

struct SQLTypeInference {
    static mlir::FloatType get_higher_float_type(mlir::Type left, mlir::Type right) {
        auto left_float = dyn_cast_or_null<mlir::FloatType>(left);
        if (auto right_float = dyn_cast_or_null<mlir::FloatType>(right)) {
            if (!left_float || right_float.getWidth() > left_float.getWidth()) {
                return right_float;
            }
        }
        return left_float;
    }
    static mlir::IntegerType get_higher_int_type(mlir::Type left, mlir::Type right) {
        const mlir::IntegerType LEFT_INT = dyn_cast_or_null<mlir::IntegerType>(left);
        if (const auto RIGHT_INT = dyn_cast_or_null<mlir::IntegerType>(right)) {
            if (!LEFT_INT || RIGHT_INT.getWidth() > LEFT_INT.getWidth()) {
                return RIGHT_INT;
            }
        }
        return LEFT_INT;
    }
    static mlir::db::DecimalType get_higher_decimal_type(mlir::Type left, mlir::Type right) {
        const auto A = dyn_cast_or_null<mlir::db::DecimalType>(left);
        if (const auto B = dyn_cast_or_null<mlir::db::DecimalType>(right)) {
            if (!A) {
                return B;
}
            const int HIDIG = std::max(A.getP() - A.getS(), B.getP() - B.getS());
            const int MAXS = std::max(A.getS(), B.getS());
            return mlir::db::DecimalType::get(A.getContext(), std::min(HIDIG + MAXS, MAX_NUMERIC_PRECISION),
                                              std::min(MAXS, MAX_NUMERIC_UNCONSTRAINED_SCALE));
        }
        return A;
    }
    static mlir::Value cast_value_to_type(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
        const bool IS_NULLABLE = isa<mlir::db::NullableType>(v.getType());
        if (IS_NULLABLE && !isa<mlir::db::NullableType>(t)) {
            t = mlir::db::NullableType::get(builder.getContext(), t);
        }
        const bool ONLY_TARGET_IS_NULLABLE = !IS_NULLABLE && isa<mlir::db::NullableType>(t);
        if (v.getType() == t) {
            return v;
        }
        if (auto* def_op = v.getDefiningOp()) {
            if (auto const_op = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(def_op)) {
                if (!isa<mlir::db::NullableType>(t)) {
                    const_op.getResult().setType(t);
                    return const_op;
                }
            }
            if (auto null_op = mlir::dyn_cast_or_null<mlir::db::NullOp>(def_op)) {
                if (null_op.getResult().getType() == t) { // This was changed from lingodb, unsure if it's going to be
                                                         // problematic...
                    return null_op;
                }
                return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), t);
            }
        }
        if (v.getType() == getBaseType(t)) {
            return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, v);
        }
        if (ONLY_TARGET_IS_NULLABLE) {
            mlir::Value const casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), getBaseType(t), v);
            return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, casted);
        }             return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
       
    }
    static mlir::Type get_common_base_type(mlir::Type left, mlir::Type right) {
        left = getBaseType(left);
        right = getBaseType(right);

        const bool LEFT_IS_DATE = isa<mlir::db::DateType>(left);
        const bool RIGHT_IS_DATE = isa<mlir::db::DateType>(right);
        const bool LEFT_IS_TIMESTAMP = isa<mlir::db::TimestampType>(left);
        const bool RIGHT_IS_TIMESTAMP = isa<mlir::db::TimestampType>(right);

        if ((LEFT_IS_DATE || LEFT_IS_TIMESTAMP) && (RIGHT_IS_DATE || RIGHT_IS_TIMESTAMP)) {
            if (LEFT_IS_TIMESTAMP) {
                return left;
}
            if (RIGHT_IS_TIMESTAMP) {
                return right;
}
            return left;
        }

        const bool STRING_PRESENT = isa<mlir::db::StringType>(left) || isa<mlir::db::StringType>(right);
        const bool INT_PRESENT = isa<mlir::IntegerType>(left) || isa<mlir::IntegerType>(right);
        const bool FLOAT_PRESENT = isa<mlir::FloatType>(left) || isa<mlir::FloatType>(right);
        const bool DECIMAL_PRESENT = isa<mlir::db::DecimalType>(left) || isa<mlir::db::DecimalType>(right);
        if (STRING_PRESENT) {
            return mlir::db::StringType::get(left.getContext());
}
        if (DECIMAL_PRESENT) {
            return get_higher_decimal_type(left, right);
}
        if (FLOAT_PRESENT) {
            return static_cast<mlir::Type>(get_higher_float_type(left, right));
}
        if (INT_PRESENT) {
            return get_higher_int_type(left, right);
}
        return left;
    }
    static mlir::Type get_common_type(const mlir::Type LEFT, const mlir::Type RIGHT) {
        const bool IS_NULLABLE = isa<mlir::db::NullableType>(LEFT) || isa<mlir::db::NullableType>(RIGHT);
        const auto commonBaseType = get_common_base_type(LEFT, RIGHT);
        if (IS_NULLABLE) {
            return mlir::db::NullableType::get(LEFT.getContext(), commonBaseType);
        }             return commonBaseType;
       
    }
    static mlir::Type get_common_base_type(const mlir::TypeRange TYPES) {
        mlir::Type common_type = TYPES.front();
        for (const auto T : TYPES) {
            common_type = get_common_base_type(common_type, T);
        }
        return common_type;
    }
    static std::vector<mlir::Value> to_common_base_types(mlir::OpBuilder& builder, const mlir::ValueRange VALUES) {
        const auto COMMON_TYPE = get_common_base_type(VALUES.getTypes());
        std::vector<mlir::Value> res;
        for (const auto VAL : VALUES) {
            res.push_back(cast_value_to_type(builder, VAL, COMMON_TYPE));
        }
        return res;
    }
    static std::vector<mlir::Value>
    to_common_base_types_except_decimals(mlir::OpBuilder& builder, const mlir::ValueRange VALUES) {
        std::vector<mlir::Value> res;
        for (auto val : VALUES) {
            if (!isa<mlir::db::DecimalType>(getBaseType(val.getType()))) {
                return to_common_base_types(builder, VALUES);
            }
            res.push_back(val);
        }
        return res;
    }
};

auto PostgreSQLASTTranslator::Impl::upcast_binary_operation(const QueryCtxT& ctx, const mlir::Value LHS,
                                                            const mlir::Value RHS)
    -> std::pair<mlir::Value, mlir::Value> {
    auto convert_to_type = [&ctx](mlir::Value value, mlir::Type target_base_type, const bool NEEDS_NULLABLE) -> mlir::Value {
        const auto CURRENT_TYPE = value.getType();
        const auto CURRENT_BASE_TYPE = getBaseType(CURRENT_TYPE);
        const bool IS_NULLABLE = mlir::isa<mlir::db::NullableType>(CURRENT_TYPE);
        const auto LOC = ctx.builder.getUnknownLoc();

        if (CURRENT_BASE_TYPE != target_base_type) {
            if (IS_NULLABLE) {
                const auto TARGET_TYPE = mlir::db::NullableType::get(ctx.builder.getContext(), target_base_type);
                value = ctx.builder.create<mlir::db::CastOp>(LOC, TARGET_TYPE, value);
            } else {
                value = ctx.builder.create<mlir::db::CastOp>(LOC, target_base_type, value);
            }
        }

        if (NEEDS_NULLABLE && !mlir::isa<mlir::db::NullableType>(value.getType())) {
            const auto NULLABLE_TYPE = mlir::db::NullableType::get(ctx.builder.getContext(), getBaseType(value.getType()));
            value = ctx.builder.create<mlir::db::AsNullableOp>(LOC, NULLABLE_TYPE, value);
        }

        return value;
    };

    const auto LHS_BASE_TYPE = getBaseType(LHS.getType());
    const auto RHS_BASE_TYPE = getBaseType(RHS.getType());
    const auto TARGET_BASE_TYPE = (LHS_BASE_TYPE != RHS_BASE_TYPE)
                                    ? SQLTypeInference::get_common_base_type(LHS_BASE_TYPE, RHS_BASE_TYPE)
                                    : LHS_BASE_TYPE;

    const bool NEEDS_NULLABLE = mlir::isa<mlir::db::NullableType>(LHS.getType())
                               || mlir::isa<mlir::db::NullableType>(RHS.getType());

    auto converted_lhs = convert_to_type(LHS, TARGET_BASE_TYPE, NEEDS_NULLABLE);
    auto converted_rhs = convert_to_type(RHS, TARGET_BASE_TYPE, NEEDS_NULLABLE);

    return {converted_lhs, converted_rhs};
}

auto PostgreSQLASTTranslator::Impl::verify_and_print(const mlir::Value VAL) -> void {
#ifndef PGX_RELEASE_MODE
    PGX_IO(AST_TRANSLATE);
    if (auto* def_op = VAL.getDefiningOp()) {
        const auto VERIFY_RESULT = mlir::verify(def_op);
        if (mlir::failed(VERIFY_RESULT)) {
            PGX_ERROR("MLIR verification FAILED for value");
            throw std::runtime_error("MLIR verification FAILED for value");
        }
    } else {
        PGX_LOG(AST_TRANSLATE, TRACE, "val had no defining op");
    }

    PGX_LOG(AST_TRANSLATE, TRACE, "finished verification - now printing.");
    try {
        std::string value_str;
        llvm::raw_string_ostream stream(value_str);
        VAL.print(stream);
        stream.flush();
        if (value_str.empty()) {
            PGX_LOG(AST_TRANSLATE, TRACE, "<empty print output>");
        } else {
            PGX_LOG(AST_TRANSLATE, TRACE, "%s", value_str.c_str());
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during value print: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception during value print");
    }
#endif
}

auto PostgreSQLASTTranslator::Impl::print_type(const mlir::Type VAL) -> void {
    std::string value_str;
    llvm::raw_string_ostream stream(value_str);
    VAL.print(stream);
    stream.flush();
    PGX_LOG(AST_TRANSLATE, TRACE, "%s", value_str.c_str());
}

auto PostgreSQLASTTranslator::Impl::translate_comparison_op(const QueryCtxT& ctx, const Oid OP_OID,
                                                            const mlir::Value LHS, const mlir::Value RHS) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!LHS || !RHS) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: nullptr operands for OID %d", OP_OID);
        throw std::runtime_error("invalid state");
    }

    char* const oprname = get_opname(OP_OID);
    if (!oprname) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Failed to get operator name for OID %d", OP_OID);
        throw std::runtime_error("invalid state");
    }

    const std::string OP(oprname);
    pfree(oprname);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Processing operator '%s' (OID %d)", OP.c_str(), OP_OID);

    {
        print_type(LHS.getType());
        print_type(RHS.getType());
    }

    mlir::db::DBCmpPredicate predicate;
    if (OP == "=") {
        predicate = mlir::db::DBCmpPredicate::eq;
    } else if (OP == "<>" || OP == "!=") {
        predicate = mlir::db::DBCmpPredicate::neq;
    } else if (OP == "<") {
        predicate = mlir::db::DBCmpPredicate::lt;
    } else if (OP == "<=") {
        predicate = mlir::db::DBCmpPredicate::lte;
    } else if (OP == ">") {
        predicate = mlir::db::DBCmpPredicate::gt;
    } else if (OP == ">=") {
        predicate = mlir::db::DBCmpPredicate::gte;
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Unhandled operator: %s (OID: %d)", OP.c_str(), OP_OID);
        return nullptr;
    }

    auto [convertedLhs, convertedRhs] = upcast_binary_operation(ctx, LHS, RHS);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_comparison_op: Creating CmpOp with predicate %d",
            static_cast<int>(predicate));
    return ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}
} // namespace postgresql_ast