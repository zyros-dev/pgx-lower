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

mlir::Value PostgreSQLASTTranslator::Impl::translate_coerce_via_io(const QueryCtxT& ctx, Expr* expr,
                                                                   OptRefT<const TranslationResult> current_result) {
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
    for (auto elemValue : arrayElements) {
        auto normalizedLeft = leftValue;
        auto normalizedElem = elemValue;

        // Check if we're dealing with string types that might need normalization
        auto get_base_type = [](mlir::Type t) -> mlir::Type {
            if (auto nullable = mlir::dyn_cast<mlir::db::NullableType>(t)) {
                return nullable.getType();
            }
            return t;
        };

        const bool left_is_string = mlir::isa<mlir::db::StringType>(get_base_type(normalizedLeft.getType()));
        const bool elem_is_string = mlir::isa<mlir::db::StringType>(get_base_type(normalizedElem.getType()));

        if (left_is_string && elem_is_string) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "String comparison in array operation - BPCHAR normalization may apply");
        }

        mlir::Value cmp = nullptr;

        if (op == "=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      normalizedLeft, normalizedElem);
        } else if (op == "<>" || op == "!=") {
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::neq,
                                                      normalizedLeft, normalizedElem);
        } else {
            PGX_WARNING("Unsupported operator '%s' in ScalarArrayOpExpr, defaulting to equality", op.c_str());
            cmp = ctx.builder.create<mlir::db::CmpOp>(ctx.builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq,
                                                      normalizedLeft, normalizedElem);
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

auto PostgreSQLASTTranslator::Impl::translate_case_expr(const QueryCtxT& ctx, const CaseExpr* case_expr,
                                                        OptRefT<const TranslationResult> current_result) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!case_expr) {
        PGX_ERROR("Invalid CaseExpr parameters");
        throw std::runtime_error("Check logs");
    }

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

    if (nodeTag(expr) == T_CaseTestExpr) {
        return case_test_value;
    }

    if (nodeTag(expr) == T_OpExpr) {
        const auto opExpr = reinterpret_cast<OpExpr*>(expr);

        if (!opExpr->args || opExpr->args->length != 2) {
            PGX_ERROR("OpExpr in CASE requires exactly 2 arguments");
            throw std::runtime_error("OpExpr in CASE requires exactly 2 arguments");
        }

        const auto leftNode = static_cast<Node*>(lfirst(&opExpr->args->elements[0]));
        const auto rightNode = static_cast<Node*>(lfirst(&opExpr->args->elements[1]));

        mlir::Value leftValue = (leftNode && nodeTag(leftNode) == T_CaseTestExpr)
                                    ? case_test_value
                                    : translate_expression(ctx, reinterpret_cast<Expr*>(leftNode), std::nullopt);
        mlir::Value rightValue = (rightNode && nodeTag(rightNode) == T_CaseTestExpr)
                                     ? case_test_value
                                     : translate_expression(ctx, reinterpret_cast<Expr*>(rightNode), std::nullopt);

        if (!leftValue || !rightValue) {
            PGX_ERROR("Failed to translate operands in CASE OpExpr");
            throw std::runtime_error("Failed to translate operands in CASE OpExpr");
        }

        std::tie(leftValue, rightValue) = normalize_bpchar_operands(ctx, opExpr, leftValue, rightValue);

        return translate_comparison_op(ctx, opExpr->opno, leftValue, rightValue);
    }

    return translate_expression(ctx, expr);
}

} // namespace postgresql_ast