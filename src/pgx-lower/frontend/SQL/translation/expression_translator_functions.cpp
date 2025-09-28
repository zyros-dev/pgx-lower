#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
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

auto PostgreSQLASTTranslator::Impl::translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                                        OptRefT<const TranslationResult> current_result,
                                                        std::optional<std::vector<mlir::Value>> pre_translated_args)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!func_expr) {
        PGX_ERROR("Invalid FuncExpr parameters");
        throw std::runtime_error("Invalid FuncExpr parameters");
    }

    auto args = std::vector<mlir::Value>{};
    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_func_expr: has_pre_translated_args=%d, has_current_result=%d",
            pre_translated_args.has_value(), current_result.has_value());
    if (pre_translated_args) {
        args = *pre_translated_args;
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_func_expr: Using pre_translated_args with %zu args",
                args.size());
    } else if (func_expr->args && func_expr->args->length > 0) {
        if (!func_expr->args->elements) {
            PGX_ERROR("FuncExpr args list has length but no elements array");
            throw std::runtime_error("FuncExpr args list has length but no elements array");
        }

        ListCell* lc;
        foreach (lc, func_expr->args) {
            if (const auto argNode = static_cast<Node*>(lfirst(lc))) {
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "[SCOPE_DEBUG] translate_func_expr: About to translate arg, has_current_result=%d",
                        current_result.has_value());
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
            targetBaseType = static_cast<mlir::Type>(ctx.builder.getF32Type());
        } else if (func == "float8") {
            targetBaseType = static_cast<mlir::Type>(ctx.builder.getF64Type());
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

auto PostgreSQLASTTranslator::Impl::translate_func_expr_with_join_context(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                                                          const TranslationResult* left_child,
                                                                          const TranslationResult* right_child)
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
                                                                                  left_child, right_child))
                {
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
        auto runtimeCall = ctx.builder.create<mlir::db::RuntimeCall>(loc, ctx.builder.getI64Type(), "ExtractFromDate",
                                                                     args);
        return runtimeCall.getRes();
    }

    return translate_func_expr(ctx, func_expr, std::nullopt, args);
}

auto PostgreSQLASTTranslator::Impl::translate_subplan(const QueryCtxT& ctx, const SubPlan* subplan,
                                                      OptRefT<const TranslationResult> current_result) -> mlir::Value {
    switch (subplan->subLinkType) {
    case EXPR_SUBLINK: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "EXPR_SUBLINK: Translating scalar subquery");

        // Extract subquery plan
        if (subplan->plan_id < 1 || subplan->plan_id > list_length(ctx.current_stmt.subplans)) {
            PGX_ERROR("Invalid plan_id=%d (subplans count=%d)", subplan->plan_id, list_length(ctx.current_stmt.subplans));
            throw std::runtime_error("Invalid SubPlan plan_id");
        }

        auto subquery_plan = static_cast<Plan*>(list_nth(ctx.current_stmt.subplans, subplan->plan_id - 1));

        // Set up correlation parameters from subplan->parParam and subplan->args
        std::unordered_map<int, std::pair<std::string, std::string>> correlation_mapping;
        if (subplan->parParam && subplan->args) {
            int num_params = list_length(subplan->parParam);
            for (int i = 0; i < num_params; i++) {
                int param_id = lfirst_int(list_nth_cell(subplan->parParam, i));
                auto arg_expr = static_cast<Expr*>(lfirst(list_nth_cell(subplan->args, i)));

                if (arg_expr && nodeTag(arg_expr) == T_Var) {
                    auto var = reinterpret_cast<Var*>(arg_expr);
                    std::string table_scope;
                    std::string column_name;

                    if (var->varno == INNER_VAR || var->varno == OUTER_VAR) {
                        if (current_result) {
                            if (auto resolved = current_result->get().resolve_var(var->varno, var->varattno)) {
                                table_scope = resolved->first;
                                column_name = resolved->second;
                                PGX_LOG(AST_TRANSLATE, DEBUG,
                                        "Resolved INNER_VAR/OUTER_VAR via varno_resolution: varno=%d, varattno=%d -> "
                                        "%s.%s",
                                        var->varno, var->varattno, table_scope.c_str(), column_name.c_str());
                            } else if (var->varno == OUTER_VAR && var->varattno > 0
                                       && var->varattno <= static_cast<int>(current_result->get().columns.size()))
                            {
                                const auto& col = current_result->get().columns[var->varattno - 1];
                                table_scope = col.table_name;
                                column_name = col.column_name;
                                PGX_LOG(AST_TRANSLATE, DEBUG, "Resolved OUTER_VAR via columns: varattno=%d -> %s.%s",
                                        var->varattno, table_scope.c_str(), column_name.c_str());
                            } else {
                                PGX_ERROR("Cannot resolve INNER_VAR/OUTER_VAR: varno=%d, varattno=%d", var->varno,
                                          var->varattno);
                                throw std::runtime_error("Cannot resolve join variable in correlation parameter");
                            }
                        } else {
                            PGX_ERROR("INNER_VAR/OUTER_VAR in correlation but no current_result available");
                            throw std::runtime_error("Cannot resolve join variable without context");
                        }
                    } else {
                        table_scope = get_table_alias_from_rte(&ctx.current_stmt, var->varno);
                        column_name = get_column_name_from_schema(&ctx.current_stmt, var->varno, var->varattno);
                    }

                    correlation_mapping[param_id] = {table_scope, column_name};
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Mapped correlation paramid=%d to %s.%s", param_id,
                            table_scope.c_str(), column_name.c_str());
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
            ctx.builder.getUnknownLoc(), result_type, column_ref, subquery_stream);

        PGX_LOG(AST_TRANSLATE, DEBUG, "EXPR_SUBLINK: Created GetScalarOp for %s.%s", result_column.table_name.c_str(),
                result_column.column_name.c_str());

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

        auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), subquery_stream);

        auto& pred_region = selection_op.getPredicate();
        auto& pred_block = pred_region.emplaceBlock();
        auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

        mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

        auto inner_ctx = QueryCtxT(ctx.current_stmt, pred_builder, ctx.current_module, inner_tuple, mlir::Value());
        inner_ctx.init_plan_results = ctx.init_plan_results;
        inner_ctx.nest_params = ctx.nest_params;
        inner_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
        inner_ctx.correlation_params = ctx.correlation_params;

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

                    PGX_LOG(AST_TRANSLATE, DEBUG, "  Mapped paramId=%d to column %s.%s (index %d)", param_id,
                            column_schema.table_name.c_str(), column_schema.column_name.c_str(), i);
                } else {
                    PGX_ERROR("ParamId=%d index %d exceeds subquery columns (%zu)", param_id, i,
                              subquery_result.columns.size());
                    throw std::runtime_error("ParamId index out of range");
                }
            }
        }

        auto comparison = translate_expression(inner_ctx, reinterpret_cast<Expr*>(subplan->testexpr), current_result);
        if (!comparison) {
            PGX_ERROR("Failed to translate ANY_SUBLINK testexpr");
            throw std::runtime_error("Failed to translate testexpr");
        }

        pred_builder.create<mlir::relalg::ReturnOp>(pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

        auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(ctx.builder.getUnknownLoc(),
                                                                    ctx.builder.getI1Type(), selection_op.getResult());

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

        auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(), subquery_stream);

        auto& pred_region = selection_op.getPredicate();
        auto& pred_block = pred_region.emplaceBlock();
        auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

        mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

        auto inner_ctx = QueryCtxT(ctx.current_stmt, pred_builder, ctx.current_module, inner_tuple, mlir::Value());
        inner_ctx.init_plan_results = ctx.init_plan_results;
        inner_ctx.nest_params = ctx.nest_params;
        inner_ctx.subquery_param_mapping = ctx.subquery_param_mapping;
        inner_ctx.correlation_params = ctx.correlation_params;

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

                    PGX_LOG(AST_TRANSLATE, DEBUG, "  Mapped paramId=%d to column %s.%s (index %d)", param_id,
                            column_schema.table_name.c_str(), column_schema.column_name.c_str(), i);
                } else {
                    PGX_ERROR("ParamId=%d index %d exceeds subquery columns (%zu)", param_id, i,
                              subquery_result.columns.size());
                    throw std::runtime_error("ParamId index out of range");
                }
            }
        }

        auto comparison = translate_expression(inner_ctx, reinterpret_cast<Expr*>(subplan->testexpr), current_result);
        if (!comparison) {
            PGX_ERROR("Failed to translate ALL_SUBLINK testexpr");
            throw std::runtime_error("Failed to translate testexpr");
        }

        auto negated_comparison = pred_builder.create<mlir::db::NotOp>(pred_builder.getUnknownLoc(),
                                                                       comparison.getType(), comparison);

        pred_builder.create<mlir::relalg::ReturnOp>(pred_builder.getUnknownLoc(),
                                                    mlir::ValueRange{negated_comparison.getResult()});

        auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(ctx.builder.getUnknownLoc(),
                                                                    ctx.builder.getI1Type(), selection_op.getResult());

        auto final_not = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), exists_op.getType(),
                                                             exists_op.getResult());

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

auto PostgreSQLASTTranslator::Impl::translate_subquery_plan(const QueryCtxT& parent_ctx, Plan* subquery_plan,
                                                            const PlannedStmt* parent_stmt)
    -> std::pair<mlir::Value, TranslationResult> {
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

    auto subquery_ctx = QueryCtxT(*parent_stmt, parent_ctx.builder, parent_ctx.current_module, parent_ctx.current_tuple,
                                  mlir::Value());
    subquery_ctx.init_plan_results = parent_ctx.init_plan_results;
    subquery_ctx.nest_params = parent_ctx.nest_params;
    subquery_ctx.subquery_param_mapping = parent_ctx.subquery_param_mapping;
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

    if (nodeTag(expr) == T_Var) {
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

    auto realBlockCtx = QueryCtxT{ctx.current_stmt, realBlockBuilder, ctx.current_module, realTupleArg,
                                  ctx.current_tuple};
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

auto PostgreSQLASTTranslator::Impl::translate_expression_with_join_context(const QueryCtxT& ctx, Expr* expr,
                                                                           const TranslationResult* left_child,
                                                                           const TranslationResult* right_child)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!expr) {
        PGX_ERROR("Null expression");
        throw std::runtime_error("check logs");
    }

    if (nodeTag(expr) == T_Var) {
        const auto* var = reinterpret_cast<const Var*>(expr);

        PGX_LOG(AST_TRANSLATE, DEBUG,
                "[VAR RESOLUTION] Processing Var: varno=%d, varattno=%d, varattnosyn=%d, vartype=%d", var->varno,
                var->varattno, var->varattnosyn, var->vartype);

        if (var->varno == OUTER_VAR && left_child) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] OUTER_VAR: left_child has %zu columns",
                    left_child->columns.size());
            for (size_t i = 0; i < left_child->columns.size(); ++i) {
                const auto& c = left_child->columns[i];
                PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s (oid=%d)", i + 1, c.table_name.c_str(),
                        c.column_name.c_str(), c.type_oid);
            }

            if (var->varattno > 0 && var->varattno <= static_cast<int>(left_child->columns.size())) {
                const auto& col = left_child->columns[var->varattno - 1];

                PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] OUTER_VAR varattno=%d resolved to %s.%s (position %d)",
                        var->varattno, col.table_name.c_str(), col.column_name.c_str(), var->varattno - 1);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Column details: type_oid=%d, nullable=%d", col.type_oid,
                        col.nullable);

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
                PGX_ERROR("OUTER_VAR varattno %d out of range (have %zu columns)", var->varattno,
                          left_child->columns.size());
                throw std::runtime_error("Check logs");
            }
        } else if (var->varno == INNER_VAR && right_child) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INNER_VAR: right_child has %zu columns",
                    right_child->columns.size());
            for (size_t i = 0; i < right_child->columns.size(); ++i) {
                const auto& c = right_child->columns[i];
                PGX_LOG(AST_TRANSLATE, DEBUG, "  [%zu] %s.%s (oid=%d)", i + 1, c.table_name.c_str(),
                        c.column_name.c_str(), c.type_oid);
            }

            const auto* resolved_col_ptr = [&]() -> const TranslationResult::ColumnSchema* {
                if (auto mapping = right_child->resolve_var(var->varnosyn, var->varattno)) {
                    const auto& [table_name, col_name] = *mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "[VAR RESOLUTION] INNER_VAR using varno_resolution: varnosyn=%d, varattno=%d -> @%s::@%s",
                            var->varnosyn, var->varattno, table_name.c_str(), col_name.c_str());
                    for (const auto& c : right_child->columns) {
                        if (c.table_name == table_name && c.column_name == col_name) {
                            return &c;
                        }
                    }
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "[VAR RESOLUTION] Mapping found but column not in schema, falling back to position");
                }
                return nullptr;
            }();

            const auto& col = resolved_col_ptr ? *resolved_col_ptr
                              : (var->varattno > 0 && var->varattno <= static_cast<int>(right_child->columns.size()))
                                  ? right_child->columns[var->varattno - 1]
                                  : throw std::runtime_error("INNER_VAR varattno out of range");

            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INNER_VAR varattno=%d resolved to %s.%s", var->varattno,
                    col.table_name.c_str(), col.column_name.c_str());
            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] Column details: type_oid=%d, nullable=%d", col.type_oid,
                    col.nullable);

            auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
            if (!dialect) {
                PGX_ERROR("RelAlg dialect not registered");
                throw std::runtime_error("Check logs");
            }

            auto& columnManager = dialect->getColumnManager();
            auto colRef = columnManager.createRef(col.table_name, col.column_name);
            colRef.getColumn().type = col.mlir_type;

            auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), col.mlir_type,
                                                                          colRef, ctx.current_tuple);

            return getColOp.getRes();
        } else if (var->varno == INDEX_VAR && right_child) {
            PGX_LOG(AST_TRANSLATE, DEBUG,
                    "[VAR RESOLUTION] INDEX_VAR: varnosyn=%d, varattno=%d, right_child has %zu columns", var->varnosyn,
                    var->varattno, right_child->columns.size());

            const auto* resolved_col_ptr = [&]() -> const TranslationResult::ColumnSchema* {
                if (auto mapping = right_child->resolve_var(var->varnosyn, var->varattno)) {
                    const auto& [table_name, col_name] = *mapping;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "[VAR RESOLUTION] INDEX_VAR using varno_resolution: varnosyn=%d, varattno=%d -> @%s::@%s",
                            var->varnosyn, var->varattno, table_name.c_str(), col_name.c_str());
                    for (const auto& c : right_child->columns) {
                        if (c.table_name == table_name && c.column_name == col_name) {
                            return &c;
                        }
                    }
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "[VAR RESOLUTION] Mapping found but column not in schema, falling back to position");
                }
                return nullptr;
            }();

            const auto& col = resolved_col_ptr ? *resolved_col_ptr
                              : (var->varattno > 0 && var->varattno <= static_cast<int>(right_child->columns.size()))
                                  ? right_child->columns[var->varattno - 1]
                                  : throw std::runtime_error("INDEX_VAR varattno out of range");

            PGX_LOG(AST_TRANSLATE, DEBUG, "[VAR RESOLUTION] INDEX_VAR resolved to %s.%s", col.table_name.c_str(),
                    col.column_name.c_str());

            auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
            if (!dialect) {
                PGX_ERROR("RelAlg dialect not registered");
                throw std::runtime_error("Check logs");
            }

            auto& columnManager = dialect->getColumnManager();
            auto colRef = columnManager.createRef(col.table_name, col.column_name);
            colRef.getColumn().type = col.mlir_type;

            auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), col.mlir_type,
                                                                          colRef, ctx.current_tuple);

            return getColOp.getRes();
        } else {
            return translate_var(ctx, var);
        }
    }

    // For non-var, rely on the expression translator
    switch (expr->type) {
    case T_OpExpr: {
        const auto* op_expr = reinterpret_cast<const OpExpr*>(expr);
        return translate_op_expr_with_join_context(ctx, op_expr, left_child, right_child);
    }
    case T_BoolExpr: {
        const auto* bool_expr = reinterpret_cast<const BoolExpr*>(expr);
        return translate_bool_expr_with_join_context(ctx, bool_expr, left_child, right_child);
    }
    case T_FuncExpr: {
        const auto* func_expr = reinterpret_cast<const FuncExpr*>(expr);
        return translate_func_expr_with_join_context(ctx, func_expr, left_child, right_child);
    }
    case T_Param: {
        const auto* param = reinterpret_cast<const Param*>(expr);

        if (param->paramkind == PARAM_EXEC && !ctx.nest_params.empty()) {
            const auto nest_param_it = ctx.nest_params.find(param->paramid);
            if (nest_param_it != ctx.nest_params.end()) {
                auto* paramVar = nest_param_it->second;
                PGX_LOG(AST_TRANSLATE, DEBUG,
                        "Param paramid=%d references nestParam Var(varno=%d, varattno=%d) - translating in join "
                        "context",
                        param->paramid, paramVar->varno, paramVar->varattno);
                return translate_expression_with_join_context(ctx, reinterpret_cast<Expr*>(paramVar), left_child,
                                                              right_child);
            }
        }

        return translate_expression(ctx, expr);
    }
    default: {
        if (left_child || right_child) {
            TranslationResult merged_result;
            if (left_child) {
                merged_result.columns.insert(merged_result.columns.end(), left_child->columns.begin(),
                                             left_child->columns.end());
                merged_result.varno_resolution.insert(left_child->varno_resolution.begin(),
                                                      left_child->varno_resolution.end());
                for (size_t i = 0; i < left_child->columns.size(); ++i) {
                    const auto& col = left_child->columns[i];
                    merged_result.varno_resolution[std::make_pair(OUTER_VAR, i + 1)] = std::make_pair(col.table_name,
                                                                                                      col.column_name);
                }
            }
            if (right_child) {
                merged_result.columns.insert(merged_result.columns.end(), right_child->columns.begin(),
                                             right_child->columns.end());
                merged_result.varno_resolution.insert(right_child->varno_resolution.begin(),
                                                      right_child->varno_resolution.end());
                for (size_t i = 0; i < right_child->columns.size(); ++i) {
                    const auto& col = right_child->columns[i];
                    merged_result.varno_resolution[std::make_pair(INNER_VAR, i + 1)] = std::make_pair(col.table_name,
                                                                                                      col.column_name);
                }
            }
            return translate_expression(ctx, expr, merged_result);
        }
        return translate_expression(ctx, expr);
    }
    }
}

} // namespace postgresql_ast