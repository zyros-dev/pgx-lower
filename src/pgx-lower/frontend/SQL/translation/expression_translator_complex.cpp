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

namespace mlir::relalg {
class GetColumnOp;
} // namespace mlir::relalg

namespace postgresql_ast {
using namespace pgx_lower::frontend::sql::constants;

mlir::Value PostgreSQLASTTranslator::Impl::translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr) {
    const auto* coerce = reinterpret_cast<CoerceViaIO*>(expr);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Processing T_CoerceViaIO to type OID %d", coerce->resulttype);

    auto arg_value = translate_expression(ctx, coerce->arg);
    if (!arg_value) {
        PGX_ERROR("Failed to translate CoerceViaIO argument");
        throw std::runtime_error("Failed to translate CoerceViaIO argument");
    }

    const bool IS_NULLABLE = mlir::isa<mlir::db::NullableType>(arg_value.getType());
    const auto TYPE_MAPPER = PostgreSQLTypeMapper(context_);
    auto target_type = TYPE_MAPPER.map_postgre_sqltype(coerce->resulttype, -1, IS_NULLABLE);

    return ctx.builder.create<mlir::db::CastOp>(ctx.builder.getUnknownLoc(), target_type, arg_value);
}

auto PostgreSQLASTTranslator::Impl::translate_bool_expr(const QueryCtxT& ctx, const BoolExpr* bool_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!bool_expr || !bool_expr->args || bool_expr->args->length == 0) {
        PGX_ERROR("Invalid BoolExpr parameters");
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

            ListCell* lc = nullptr;
            foreach (lc, bool_expr->args) {
                if (auto *const ARG_NODE = static_cast<Node*>(lfirst(lc))) {
                    if (mlir::Value arg_value = translate_expression(ctx, reinterpret_cast<Expr*>(ARG_NODE))) {
                        if (!arg_value.getType().isInteger(1)) {
                            arg_value = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), arg_value);
                        }

                        if (!result) {
                            result = arg_value;
                        } else {
                            result = ctx.builder.create<mlir::db::AndOp>(
                                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), mlir::ValueRange{result, arg_value});
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

            ListCell* lc = nullptr;
            foreach (lc, bool_expr->args) {
                if (auto *const ARG_NODE = static_cast<Node*>(lfirst(lc))) {
                    if (auto arg_value = translate_expression(ctx, reinterpret_cast<Expr*>(ARG_NODE))) {
                        if (!arg_value.getType().isInteger(1)) { // Ensur
                            arg_value = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), arg_value);
                        }

                        if (!result) {
                            result = arg_value;
                        } else {
                            result = ctx.builder.create<mlir::db::OrOp>(
                                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), mlir::ValueRange{result, arg_value});
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
        mlir::Value arg_val = nullptr;

        if (bool_expr->args && bool_expr->args->length > 0) {
            if (const ListCell* const lc = list_head(bool_expr->args)) {
                if (auto *const ARG_NODE = static_cast<Node*>(lfirst(lc))) {
                    arg_val = translate_expression(ctx, reinterpret_cast<Expr*>(ARG_NODE));
                }
            }
        }

        if (!arg_val) {
            PGX_ERROR("NOT expression has no valid argument, using placeholder");
            throw std::runtime_error("NOT expression has no valid argument, using placeholder");
        }

        if (!arg_val.getType().isInteger(1)) {
            arg_val = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), arg_val);
        }

        return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), arg_val);
    }

    default: {
        PGX_ERROR("Unknown BoolExpr type: %d", bool_expr->boolop);
        throw std::runtime_error("Unknown BoolExpr type");
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_null_test(const QueryCtxT& ctx, const NullTest* null_test) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!null_test) {
        PGX_ERROR("Invalid NullTest parameters");
        throw std::runtime_error("Invalid NullTest parameters");
    }

    auto* arg_node = reinterpret_cast<Node*>(null_test->arg);
    auto arg_val = translate_expression(ctx, reinterpret_cast<Expr*>(arg_node));
    if (!arg_val) {
        PGX_ERROR("Failed to translate NullTest argument");
        throw std::runtime_error("Failed to translate NullTest argument");
    }

    if (isa<mlir::db::NullableType>(arg_val.getType())) {
        auto isNull = ctx.builder.create<mlir::db::IsNullOp>(ctx.builder.getUnknownLoc(), arg_val);
        if (null_test->nulltesttype == PG_IS_NOT_NULL) {
            return ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), isNull);
        }             return isNull;
    } else {
        return ctx.builder.create<mlir::db::ConstantOp>(
            ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(),
            ctx.builder.getIntegerAttr(ctx.builder.getI1Type(), static_cast<int64_t>(null_test->nulltesttype == PG_IS_NOT_NULL)));
    }
}

auto PostgreSQLASTTranslator::Impl::translate_coalesce_expr(const QueryCtxT& ctx, const CoalesceExpr* coalesce_expr)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!coalesce_expr) {
        PGX_ERROR("Invalid CoalesceExpr parameters");
        throw std::runtime_error("Invalid CoalesceExpr parameters");
    }

    if (!coalesce_expr->args || coalesce_expr->args->length == 0) {
        auto null_type = mlir::db::NullableType::get(&context_, ctx.builder.getI32Type());
        return ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), null_type);
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE has %d arguments", coalesce_expr->args->length);

    auto translated_args = std::vector<mlir::Value>{};

    ListCell* cell = nullptr;
    foreach (cell, coalesce_expr->args) {
        auto *const EXPR = static_cast<Expr*>(lfirst(cell));
        if (mlir::Value const val = translate_expression(ctx, EXPR)) {
            translated_args.push_back(val);
        } else {
            PGX_ERROR("Failed to translate COALESCE argument");
            throw std::runtime_error("Failed to translate COALESCE argument");
        }
    }

    if (translated_args.empty()) {
        PGX_ERROR("All COALESCE arguments failed to translate");
        throw std::runtime_error("All COALESCE arguments failed to translate");
    }

    mlir::Type base_type = nullptr;
    for (const auto& arg : translated_args) {
        const auto ARG_TYPE = arg.getType();
        if (auto nullable_type = dyn_cast<mlir::db::NullableType>(ARG_TYPE)) {
            if (!base_type) {
                base_type = nullable_type.getType();
            }
        } else if (!base_type) {
            base_type = ARG_TYPE;
        }
    }

    // COALESCE should always produce nullable type in query contexts
    // Even when all inputs are non-nullable, the result needs nullable wrapper
    auto common_type = mlir::db::NullableType::get(&context_, base_type);
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE common type determined - forcing nullable for query context");
    for (auto& val : translated_args) {
        if (val.getType() != common_type) {
            if (!isa<mlir::db::NullableType>(val.getType())) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "Wrapping non-nullable argument to match common nullable type");
                auto false_flag = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(), 0, 1);
                val = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), common_type, val, false_flag);
            }
        }
    }

    std::function<mlir::Value(size_t)> build_coalesce_recursive = [&](const size_t INDEX) -> mlir::Value {
        const auto LOC = ctx.builder.getUnknownLoc();
        if (INDEX >= translated_args.size() - 1) {
            return translated_args.back();
        }

        auto value = translated_args[INDEX];
        auto is_null = ctx.builder.create<mlir::db::IsNullOp>(LOC, value);
        auto is_not_null = ctx.builder.create<mlir::db::NotOp>(LOC, is_null);

        auto if_op = ctx.builder.create<mlir::scf::IfOp>(LOC, common_type, is_not_null, true);

        auto& then_region = if_op.getThenRegion();
        auto* then_block = &then_region.front();
        ctx.builder.setInsertionPointToEnd(then_block);

        mlir::Value then_value = value;
        if (value.getType() != common_type && !isa<mlir::db::NullableType>(value.getType())) {
            auto false_flag = ctx.builder.create<mlir::arith::ConstantIntOp>(LOC, 0, 1);
            then_value = ctx.builder.create<mlir::db::AsNullableOp>(LOC, common_type, value, false_flag);
        }
        ctx.builder.create<mlir::scf::YieldOp>(LOC, then_value);

        auto& else_region = if_op.getElseRegion();
        auto* else_block = &else_region.front();
        ctx.builder.setInsertionPointToEnd(else_block);
        auto else_value = build_coalesce_recursive(INDEX + 1);
        ctx.builder.create<mlir::scf::YieldOp>(LOC, else_value);

        ctx.builder.setInsertionPointAfter(if_op);

        return if_op.getResult(0);
    };

    const auto RESULT = build_coalesce_recursive(0);

    const bool RESULT_IS_NULLABLE = mlir::isa<mlir::db::NullableType>(RESULT.getType());
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE final result is nullable: %d", RESULT_IS_NULLABLE);

    const auto RESULT_IS_NULLABLE_TYPE = isa<mlir::db::NullableType>(RESULT.getType());
    PGX_LOG(AST_TRANSLATE, IO, "translate_coalesce_expr OUT: MLIR Value (nullable=%d)", RESULT_IS_NULLABLE_TYPE);

    return RESULT;
}

auto PostgreSQLASTTranslator::Impl::translate_scalar_array_op_expr(const QueryCtxT& ctx,
                                                                   const ScalarArrayOpExpr* scalar_array_op)
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

    auto *const LEFT_NODE = static_cast<Node*>(lfirst(&args->elements[0]));
    auto left_value = translate_expression(ctx, reinterpret_cast<Expr*>(LEFT_NODE));
    if (!left_value) {
        PGX_ERROR("Failed to translate left operand of IN expression");
        throw std::runtime_error("Failed to translate left operand of IN expression");
    }

    Oid const left_type_oid = exprType(LEFT_NODE);
    int32 const left_type_mod = exprTypmod(LEFT_NODE);
    int bpchar_length = -1;
    if (left_type_oid == BPCHAROID && left_type_mod >= VARHDRSZ) {
        bpchar_length = left_type_mod - VARHDRSZ;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Left operand is BPCHAR with length=%d", bpchar_length);
    }

    auto *const RIGHT_NODE = static_cast<Node*>(lfirst(&args->elements[1]));

    PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr: Right operand nodeTag = %d", nodeTag(RIGHT_NODE));
    auto array_elements = std::vector<mlir::Value>{};

    if (nodeTag(RIGHT_NODE) == T_ArrayExpr) {
        auto *const ARRAY_EXPR = reinterpret_cast<ArrayExpr*>(RIGHT_NODE);
        if (const auto* elements = ARRAY_EXPR->elements) {
            ListCell* lc = nullptr;
            foreach (lc, elements) {
                auto *const ELEM_NODE = static_cast<Node*>(lfirst(lc));
                if (mlir::Value const elem_value = translate_expression(ctx, reinterpret_cast<Expr*>(ELEM_NODE))) {
                    array_elements.push_back(elem_value);
                }
            }
        }
    } else if (nodeTag(RIGHT_NODE) == T_Const) {
        if (auto *const CONST_NODE = reinterpret_cast<Const*>(RIGHT_NODE); CONST_NODE->consttype == INT4ARRAYOID) {
            auto *const ARRAY = DatumGetArrayTypeP(CONST_NODE->constvalue);
            int nitems = 0;
            Datum* values = nullptr;
            bool* nulls = nullptr;

            deconstruct_array(ARRAY, INT4OID, sizeof(int32), true, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    int32 const int_value = DatumGetInt32(values[i]);
                    auto elem_value = ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                                                    int_value, ctx.builder.getI32Type());
                    array_elements.push_back(elem_value);
                }
            }
        } else if (CONST_NODE->consttype == PG_TEXT_ARRAY_OID) {
            auto *const ARRAY = DatumGetArrayTypeP(CONST_NODE->constvalue);
            int nitems = 0;
            Datum* values = nullptr;
            bool* nulls = nullptr;

            deconstruct_array(ARRAY, TEXTOID, -1, false, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    auto *const TEXT_VALUE = DatumGetTextP(values[i]);
                    std::string const str_value(VARDATA(TEXT_VALUE), VARSIZE(TEXT_VALUE) - VARHDRSZ);

                    auto elem_value = ctx.builder.create<mlir::db::ConstantOp>(
                        ctx.builder.getUnknownLoc(), ctx.builder.getType<mlir::db::StringType>(),
                        ctx.builder.getStringAttr(str_value));
                    array_elements.push_back(elem_value);
                }
            }
        } else if (CONST_NODE->consttype == BPCHARARRAYOID) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Processing BPCHAR array (CHAR/VARCHAR), target column length=%d",
                    bpchar_length);
            auto *const ARRAY = DatumGetArrayTypeP(CONST_NODE->constvalue);
            int nitems = 0;
            Datum* values = nullptr;
            bool* nulls = nullptr;

            deconstruct_array(ARRAY, BPCHAROID, -1, false, TYPALIGN_INT, &values, &nulls, &nitems);

            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    auto *const BPCHAR_VALUE = DatumGetBpCharP(values[i]);
                    std::string str_value(VARDATA_ANY(BPCHAR_VALUE), VARSIZE_ANY_EXHDR(BPCHAR_VALUE));

                    str_value.erase(str_value.find_last_not_of(' ') + 1);

                    if (bpchar_length > 0 && str_value.length() < static_cast<size_t>(bpchar_length)) {
                        str_value.resize(bpchar_length, ' ');
                        PGX_LOG(AST_TRANSLATE, DEBUG, "BPCHAR array element[%d]: '%s' (padded to len=%d)", i,
                                str_value.c_str(), bpchar_length);
                    } else {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "BPCHAR array element[%d]: '%s' (len=%zu, no padding needed)", i,
                                str_value.c_str(), str_value.length());
                    }

                    auto elem_value = ctx.builder.create<mlir::db::ConstantOp>(
                        ctx.builder.getUnknownLoc(), ctx.builder.getType<mlir::db::StringType>(),
                        ctx.builder.getStringAttr(str_value));
                    array_elements.push_back(elem_value);
                }
            }
        } else {
            PGX_WARNING("ScalarArrayOpExpr: Unsupported const array type %u", CONST_NODE->consttype);
        }
    } else if (nodeTag(RIGHT_NODE) == T_SubPlan) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr with SubPlan operand detected (ANY/ALL/IN subquery)");
        throw std::runtime_error("UNEXPECTED: Is this possible?");
    } else if (nodeTag(RIGHT_NODE) == T_Param) {
        const auto* param = reinterpret_cast<Param*>(RIGHT_NODE);

        if (param->paramkind != PARAM_EXEC) {
            PGX_ERROR("Only PARAM_EXEC parameters are supported (got paramkind=%d)", param->paramkind);
            throw std::runtime_error("Unsupported param kind");
        }

        const auto IT = ctx.params.find(param->paramid);
        if (IT == ctx.params.end()) {
            PGX_ERROR("Param references unknown paramid=%d (not in params map)", param->paramid);
            throw std::runtime_error("Param references unknown param");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Resolving ScalarArrayOpExpr Param paramid=%d to InitPlan result", param->paramid);

        const auto& param_info = IT->second;

        if (!param_info.cached_value) {
            PGX_ERROR("InitPlan param %d has no cached value", param->paramid);
            throw std::runtime_error("Invalid InitPlan param");
        }
        TranslationResult initplan_result;
        initplan_result.op = param_info.cached_value->getDefiningOp();
        initplan_result.columns.push_back(TranslationResult::ColumnSchema{
            .table_name = param_info.table_name,
            .column_name = param_info.column_name,
            .type_oid = param_info.type_oid,
            .typmod = param_info.typmod,
            .mlir_type = param_info.mlir_type,
            .nullable = param_info.nullable
        });

        if (!initplan_result.op) {
            PGX_ERROR("InitPlan result for paramid=%d has no operation", param->paramid);
            throw std::runtime_error("Invalid InitPlan result");
        }

        if (initplan_result.columns.empty()) {
            PGX_ERROR("InitPlan result for paramid=%d has no columns", param->paramid);
            throw std::runtime_error("InitPlan must return at least one column");
        }

        mlir::Value const initplan_stream = initplan_result.op->getResult(0);
        const auto& initplan_column = initplan_result.columns[0];

        auto& col_mgr = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

        const auto TUPLE_TYPE = mlir::relalg::TupleType::get(ctx.builder.getContext());

        auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), initplan_stream);

        auto& pred_region = selection_op.getPredicate();
        auto& pred_block = pred_region.emplaceBlock();
        auto inner_tuple = pred_block.addArgument(TUPLE_TYPE, ctx.builder.getUnknownLoc());

        mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

        auto initplan_col_ref = col_mgr.createRef(initplan_column.table_name, initplan_column.column_name);
        auto initplan_value = pred_builder.create<mlir::relalg::GetColumnOp>(
            pred_builder.getUnknownLoc(), initplan_column.mlir_type, initplan_col_ref, inner_tuple);

        char* const oprname = get_opname(scalar_array_op->opno);
        if (!oprname) {
            PGX_ERROR("Unknown operator OID %u in ScalarArrayOpExpr with Param", scalar_array_op->opno);
            throw std::runtime_error("Unknown operator OID");
        }

        const std::string OP(oprname);
        pfree(oprname);

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
            PGX_ERROR("Unsupported operator '%s' in ScalarArrayOpExpr with Param", OP.c_str());
            throw std::runtime_error("Unsupported operator");
        }

        auto comparison = pred_builder.create<mlir::db::CmpOp>(pred_builder.getUnknownLoc(), predicate, left_value,
                                                               initplan_value);

        pred_builder.create<mlir::relalg::ReturnOp>(pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

        auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(ctx.builder.getUnknownLoc(),
                                                                    ctx.builder.getI1Type(), selection_op.getResult());

        PGX_LOG(AST_TRANSLATE, DEBUG, "Created EXISTS pattern for ScalarArrayOpExpr with Param");
        return exists_op.getResult();
    } else {
        PGX_ERROR("ScalarArrayOpExpr: Unexpected right operand type %d", nodeTag(RIGHT_NODE));
        throw std::runtime_error("Unsupported ScalarArrayOpExpr operand type");
    }

    if (array_elements.empty()) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Empty array in IN clause, returning %s",
                scalar_array_op->useOr ? "false" : "true");
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.builder.getUnknownLoc(),
                                                              scalar_array_op->useOr ? 0 : 1, ctx.builder.getI1Type());
    }

    char* const oprname = get_opname(scalar_array_op->opno);
    std::string const op = oprname ? std::string(oprname) : "=";
    if (oprname) {
        pfree(oprname);
}

    if (op == "=" && scalar_array_op->useOr) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using db.oneof for IN clause with %zu array elements", array_elements.size());

        std::vector<mlir::Value> values;
        values.push_back(left_value);
        values.insert(values.end(), array_elements.begin(), array_elements.end());

        auto oneof_op = ctx.builder.create<mlir::db::OneOfOp>(ctx.builder.getUnknownLoc(), values);
        PGX_LOG(AST_TRANSLATE, IO, "translate_scalar_array_op_expr OUT: db.oneof MLIR Value");
        return oneof_op.getResult();
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Using comparison loop for operator '%s' with useOr=%d", op.c_str(),
            scalar_array_op->useOr);

    mlir::Value result = nullptr;
    for (auto elem_value : array_elements) {
        auto normalized_left = left_value;
        auto normalized_elem = elem_value;

        auto get_base_type = [](mlir::Type t) -> mlir::Type {
            if (const auto NULLABLE = mlir::dyn_cast<mlir::db::NullableType>(t)) {
                return NULLABLE.getType();
            }
            return t;
        };

        const bool LEFT_IS_STRING = mlir::isa<mlir::db::StringType>(get_base_type(normalized_left.getType()));
        const bool ELEM_IS_STRING = mlir::isa<mlir::db::StringType>(get_base_type(normalized_elem.getType()));

        if (LEFT_IS_STRING && ELEM_IS_STRING) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "String comparison in array operation - BPCHAR normalization may apply");
        }

        mlir::Value cmp = nullptr;

        if (op == "=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      normalized_left, normalized_elem);
        } else if (op == "<>" || op == "!=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::neq,
                                                      normalized_left, normalized_elem);
        } else {
            PGX_WARNING("Unsupported operator '%s' in ScalarArrayOpExpr, defaulting to equality", op.c_str());
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      normalized_left, normalized_elem);
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

auto PostgreSQLASTTranslator::Impl::translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!case_expr) {
        PGX_ERROR("Invalid CaseExpr parameters");
        throw std::runtime_error("Check logs");
    }

    // 1. Simple:   CASE expr WHEN val1 THEN result1 WHEN val2 THEN result2 ELSE default END
    // 2. Searched: CASE WHEN cond1 THEN result1 WHEN cond2 THEN result2 ELSE default END
    mlir::Value case_arg = nullptr;
    if (case_expr->arg) {
        case_arg = translate_expression(ctx, case_expr->arg);
        if (!case_arg) {
            PGX_ERROR("Failed to translate CASE argument expression");
            throw std::runtime_error("Check logs");
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Simple CASE expression with comparison argument");
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Searched CASE expression (no comparison argument)");
    }

    // Build nested if-then-else structure from WHEN clauses
    mlir::Value else_result = nullptr;
    if (case_expr->defresult) {
        else_result = translate_expression(ctx, case_expr->defresult);
        if (!else_result) {
            PGX_ERROR("Failed to translate CASE ELSE expression");
            throw std::runtime_error("Check logs");
        }
    } else {
        const auto BASE_TYPE = ctx.builder.getI32Type();
        auto nullable_type = mlir::db::NullableType::get(ctx.builder.getContext(), BASE_TYPE);
        else_result = ctx.builder.create<mlir::db::NullOp>(ctx.builder.getUnknownLoc(), nullable_type);
    }

    // Process WHEN clauses in reverse order to build nested if-else chain
    mlir::Value result = else_result;
    if (case_expr->args && case_expr->args->length > 0) {
        for (int i = case_expr->args->length - 1; i >= 0; i--) {
            auto *const WHEN_NODE = static_cast<Node*>(lfirst(&case_expr->args->elements[i]));
            if (nodeTag(WHEN_NODE) != T_CaseWhen) {
                PGX_ERROR("Expected CaseWhen node in CASE args, got %d", nodeTag(WHEN_NODE));
                throw std::runtime_error("Check logs");
            }

            auto *const WHEN_CLAUSE = reinterpret_cast<CaseWhen*>(WHEN_NODE);

            mlir::Value condition = nullptr;
            if (case_arg) {
                const mlir::Value WHEN_CONDITION = translate_expression_with_case_test(ctx, WHEN_CLAUSE->expr, case_arg);
                if (!WHEN_CONDITION) {
                    PGX_ERROR("Failed to translate WHEN condition in simple CASE");
                    throw std::runtime_error("Check logs");
                }
                condition = WHEN_CONDITION;
            } else {
                condition = translate_expression(ctx, WHEN_CLAUSE->expr);
                if (!condition) {
                    PGX_ERROR("Failed to translate WHEN condition");
                    throw std::runtime_error("Check logs");
                }
            }

            if (auto condition_type = condition.getType();
                !isa<mlir::IntegerType>(condition_type) || cast<mlir::IntegerType>(condition_type).getWidth() != 1)
            {
                condition = ctx.builder.create<mlir::db::DeriveTruth>(ctx.builder.getUnknownLoc(), condition);
            }

            mlir::Value then_result = translate_expression(ctx, WHEN_CLAUSE->result);
            if (!then_result) {
                PGX_ERROR("Failed to translate THEN result");
                throw std::runtime_error("Check logs");
            }

            auto result_type = result.getType();
            if (auto then_type = then_result.getType(); result_type != then_type) {
                const bool RESULT_IS_NULLABLE = isa<mlir::db::NullableType>(result_type);

                if (const bool THEN_IS_NULLABLE = isa<mlir::db::NullableType>(then_type); RESULT_IS_NULLABLE && !THEN_IS_NULLABLE)
                {
                    auto nullable_type = mlir::db::NullableType::get(ctx.builder.getContext(), then_type);
                    then_result = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullable_type,
                                                                            then_result);
                } else if (!RESULT_IS_NULLABLE && THEN_IS_NULLABLE) {
                    auto nullable_type = mlir::db::NullableType::get(ctx.builder.getContext(), result_type);
                    result = ctx.builder.create<mlir::db::AsNullableOp>(ctx.builder.getUnknownLoc(), nullable_type,
                                                                        result);
                    result_type = nullable_type;
                }
            }

            auto if_op = ctx.builder.create<mlir::scf::IfOp>(ctx.builder.getUnknownLoc(), then_result.getType(),
                                                            condition, true);

            ctx.builder.setInsertionPointToStart(&if_op.getThenRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), then_result);

            ctx.builder.setInsertionPointToStart(&if_op.getElseRegion().front());
            ctx.builder.create<mlir::scf::YieldOp>(ctx.builder.getUnknownLoc(), result);

            ctx.builder.setInsertionPointAfter(if_op);
            result = if_op.getResult(0);
        }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translate_case_expr OUT: MLIR Value (CASE expression)");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_expression_with_case_test(const QueryCtxT& ctx, Expr* expr,
                                                                        const mlir::Value CASE_TEST_VALUE)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!expr) {
        throw std::runtime_error("Invalid expression");
    }

    if (nodeTag(expr) == T_CaseTestExpr) {
        return CASE_TEST_VALUE;
    }

    if (nodeTag(expr) == T_OpExpr) {
        auto *const OP_EXPR = reinterpret_cast<OpExpr*>(expr);

        if (!OP_EXPR->args || OP_EXPR->args->length != 2) {
            PGX_ERROR("OpExpr in CASE requires exactly 2 arguments");
            throw std::runtime_error("OpExpr in CASE requires exactly 2 arguments");
        }

        auto *const LEFT_NODE = static_cast<Node*>(lfirst(&OP_EXPR->args->elements[0]));
        auto *const RIGHT_NODE = static_cast<Node*>(lfirst(&OP_EXPR->args->elements[1]));

        mlir::Value left_value = (LEFT_NODE && nodeTag(LEFT_NODE) == T_CaseTestExpr)
                                    ? CASE_TEST_VALUE
                                    : translate_expression(ctx, reinterpret_cast<Expr*>(LEFT_NODE));
        mlir::Value right_value = (RIGHT_NODE && nodeTag(RIGHT_NODE) == T_CaseTestExpr)
                                     ? CASE_TEST_VALUE
                                     : translate_expression(ctx, reinterpret_cast<Expr*>(RIGHT_NODE));

        if (!left_value || !right_value) {
            PGX_ERROR("Failed to translate operands in CASE OpExpr");
            throw std::runtime_error("Failed to translate operands in CASE OpExpr");
        }

        std::tie(left_value, right_value) = normalize_bpchar_operands(ctx, OP_EXPR, left_value, right_value);

        return translate_comparison_op(ctx, OP_EXPR->opno, left_value, right_value);
    }

    return translate_expression(ctx, expr);
}

} // namespace postgresql_ast