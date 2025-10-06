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

namespace mlir::relalg {
class GetColumnOp;
} // namespace mlir::relalg

namespace postgresql_ast {
using namespace pgx_lower::frontend::sql::constants;


auto PostgreSQLASTTranslator::Impl::translate_expression_for_stream(const QueryCtxT& ctx, Expr* expr,
                                                                    const TranslationResult& child_result,
                                                                    const std::string& suggested_name)
    -> pgx_lower::frontend::sql::StreamExpressionResult {
    PGX_IO(AST_TRANSLATE);

    if (!expr || !child_result.op) {
        PGX_ERROR("Invalid parameters for translate_expression_for_stream");
        throw std::runtime_error("Invalid parameters for translate_expression_for_stream");
    }

    mlir::Value input_stream = child_result.op->getResult(0);
    const auto& child_columns = child_result.columns;

    auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        PGX_ERROR("RelAlg dialect not registered");
        throw std::runtime_error("RelAlg dialect not registered");
    }
    auto& columnManager = dialect->getColumnManager();

    if (nodeTag(expr) == T_Var) {
        const auto var = reinterpret_cast<Var*>(expr);

        std::string tableName;
        std::string columnName;

        // Both OUTER_VAR (-2) and regular vars should use child output positions
        if (var->varattno > 0 && var->varattno <= static_cast<int>(child_columns.size())) {
            const auto& childCol = child_columns[var->varattno - 1];
            tableName = childCol.table_name;
            columnName = childCol.column_name;
            PGX_LOG(AST_TRANSLATE, DEBUG, "Var (varno=%d) resolved to child output column %d: %s.%s", var->varno,
                    var->varattno, tableName.c_str(), columnName.c_str());
        } else {
            PGX_ERROR("Var varattno=%d out of bounds (child has %zu columns)", var->varattno, child_columns.size());
            throw std::runtime_error("Var index out of bounds in aggregate context");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Expression is already a column reference: %s.%s", tableName.c_str(),
                columnName.c_str());

        auto colRef = columnManager.createRef(tableName, columnName);

        auto nested = std::vector{mlir::FlatSymbolRefAttr::get(ctx.builder.getContext(), columnName)};
        auto symbolRef = mlir::SymbolRefAttr::get(ctx.builder.getContext(), tableName, nested);
        auto columnRefAttr = mlir::relalg::ColumnRefAttr::get(ctx.builder.getContext(), symbolRef, colRef.getColumnPtr());

        return {.stream = input_stream, .column_ref = columnRefAttr, .column_name = columnName, .table_name = tableName};
    }

    // Temp map op - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

    auto blockCtx = QueryCtxT::createChildContext(ctx, blockBuilder, tupleArg);

    // Pass through the child_result so varno_resolution is available
    auto exprValue = translate_expression(blockCtx, expr);
    verify_and_print(exprValue);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Finished translating expression");
    if (!exprValue) {
        PGX_ERROR("Failed to translate expression in MapOp");
        throw std::runtime_error("Failed to translate expression in MapOp");
    }

    mlir::Type exprType = exprValue.getType();
    blockBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mlir::ValueRange{exprValue});
    tempMapOp.erase();

    // map op - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    colDef.getColumn().type = exprType;
    auto mapOp = ctx.builder.create<mlir::relalg::MapOp>(ctx.builder.getUnknownLoc(), input_stream,
                                                         ctx.builder.getArrayAttr({colDef}));
    auto& realRegion = mapOp.getPredicate();
    auto* realBlock = new mlir::Block;
    realRegion.push_back(realBlock);

    auto realTupleArg = realBlock->addArgument(tupleType, ctx.builder.getUnknownLoc());

    mlir::OpBuilder realBlockBuilder(ctx.builder.getContext());
    realBlockBuilder.setInsertionPointToStart(realBlock);

    auto realBlockCtx = QueryCtxT::createChildContext(ctx, realBlockBuilder, realTupleArg);
    auto realExprValue = translate_expression(realBlockCtx, expr);
    verify_and_print(realExprValue);
    realBlockBuilder.create<mlir::relalg::ReturnOp>(ctx.builder.getUnknownLoc(), mlir::ValueRange{realExprValue});

    auto nested = std::vector{mlir::FlatSymbolRefAttr::get(ctx.builder.getContext(), columnName)};
    auto symbolRef = mlir::SymbolRefAttr::get(ctx.builder.getContext(), scopeName, nested);
    auto columnRef = mlir::relalg::ColumnRefAttr::get(ctx.builder.getContext(), symbolRef, colDef.getColumnPtr());

    PGX_LOG(AST_TRANSLATE, DEBUG, "Created MapOp with computed column: %s.%s", scopeName.c_str(), columnName.c_str());

    return {.stream = mapOp.getResult(), .column_ref = columnRef, .column_name = columnName, .table_name = scopeName};
}

auto PostgreSQLASTTranslator::Impl::translate_func_expr(const QueryCtxT& ctx, const FuncExpr* func_expr,
                                                        std::optional<std::vector<mlir::Value>> pre_translated_args)
    -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!func_expr) {
        PGX_ERROR("Invalid FuncExpr parameters");
        throw std::runtime_error("Invalid FuncExpr parameters");
    }

    auto args = std::vector<mlir::Value>{};
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
                if (mlir::Value argValue = translate_expression(ctx, reinterpret_cast<Expr*>(argNode))) {
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

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating function %s", func.c_str());
    if (func == "abs") {
        if (args.size() != 1) {
            PGX_ERROR("ABS requires exactly 1 argument, got %d", args.size());
            throw std::runtime_error("ABS requires exactly 1 argument");
        }

        verify_and_print(args[0]);
        print_type(args[0].getType());

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
    } else if (func == "extract") {
        if (args.size() != 2) {
            PGX_ERROR("EXTRACT requires exactly 2 arguments, got %zu", args.size());
            throw std::runtime_error("Check logs");
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating EXTRACT function to ExtractFromDate runtime call");
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

auto PostgreSQLASTTranslator::Impl::translate_subplan(const QueryCtxT& ctx, const SubPlan* subplan) -> mlir::Value {
    switch (subplan->subLinkType) {
    case EXPR_SUBLINK: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "EXPR_SUBLINK: Translating scalar subquery");
        // Extract subquery plan
        if (subplan->plan_id < 1 || subplan->plan_id > list_length(ctx.current_stmt.subplans)) {
            PGX_ERROR("Invalid plan_id=%d (subplans count=%d)", subplan->plan_id, list_length(ctx.current_stmt.subplans));
            throw std::runtime_error("Invalid SubPlan plan_id");
        }

        auto subquery_plan = static_cast<Plan*>(list_nth(ctx.current_stmt.subplans, subplan->plan_id - 1));

        struct CorrelationInfo {
            std::string table_scope;
            std::string column_name;
            bool nullable;
            Oid type_oid;
            int32 typmod;
        };
        std::unordered_map<int, CorrelationInfo> correlation_mapping;
        if (subplan->parParam && subplan->args) {
            int num_params = list_length(subplan->parParam);
            for (int i = 0; i < num_params; i++) {
                int param_id = lfirst_int(list_nth_cell(subplan->parParam, i));
                auto arg_expr = static_cast<Expr*>(lfirst(list_nth_cell(subplan->args, i)));

                if (arg_expr && nodeTag(arg_expr) == T_Var) {
                    auto var = reinterpret_cast<Var*>(arg_expr);
                    std::string table_scope;
                    std::string column_name;
                    bool nullable;
                    Oid type_oid = var->vartype;
                    int32 typmod = var->vartypmod;

                    if (IS_SPECIAL_VARNO(var->varno)) {
                        auto varnosyn_opt = std::optional<int>(var->varnosyn);
                        auto varattnosyn_opt = std::optional<int>(var->varattnosyn);

                        if (auto resolved = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt))
                        {
                            table_scope = resolved->table_name;
                            column_name = resolved->column_name;
                            nullable = resolved->nullable;
                            PGX_LOG(AST_TRANSLATE, DEBUG,
                                    "Resolved synthetic varno=%d via varno_resolution -> %s.%s (nullable=%d)",
                                    var->varno, table_scope.c_str(), column_name.c_str(), nullable);
                        } else if (var->varno == OUTER_VAR) {
                            auto& result_to_use = ctx.outer_result ? ctx.outer_result.value()
                                                                   : throw std::runtime_error("OUTER_VAR without outer_result");

                            if (var->varattno <= 0 || var->varattno > static_cast<int>(result_to_use.get().columns.size())) {
                                PGX_ERROR("OUTER_VAR varattno=%d out of range (result has %zu columns)", var->varattno,
                                          result_to_use.get().columns.size());
                                throw std::runtime_error("OUTER_VAR reference out of range");
                            }
                            const auto& col = result_to_use.get().columns[var->varattno - 1];
                            table_scope = col.table_name;
                            column_name = col.column_name;
                            nullable = col.nullable;
                            type_oid = col.type_oid;
                            typmod = col.typmod;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "OUTER_VAR varattno=%d resolved to %s.%s (nullable=%d) from outer_result",
                                    var->varattno, table_scope.c_str(), column_name.c_str(), nullable);
                        } else if (var->varno == INNER_VAR || var->varno == INDEX_VAR) {
                            PGX_ERROR("INNER_VAR/INDEX_VAR varno=%d varattno=%d not found in varno_resolution - join translation bug",
                                      var->varno, var->varattno);
                            throw std::runtime_error("INNER_VAR/INDEX_VAR requires varno_resolution mapping");
                        }
                    } else {
                        table_scope = get_table_alias_from_rte(&ctx.current_stmt, var->varno);
                        column_name = get_column_name_from_schema(&ctx.current_stmt, var->varno, var->varattno);
                        nullable = is_column_nullable(&ctx.current_stmt, var->varno, var->varattno);
                    }

                    correlation_mapping[param_id] = {table_scope, column_name, nullable, type_oid, typmod};
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Mapped correlation paramid=%d to %s.%s (nullable=%d)", param_id,
                            table_scope.c_str(), column_name.c_str(), nullable);
                } else if (arg_expr && nodeTag(arg_expr) == T_Param) {
                    auto param = reinterpret_cast<Param*>(arg_expr);
                    const auto param_it = ctx.params.find(param->paramid);

                    if (param_it != ctx.params.end()) {
                        const auto& resolved = param_it->second;
                        correlation_mapping[param_id] = {resolved.table_name, resolved.column_name, resolved.nullable,
                                                         resolved.type_oid, resolved.typmod};
                        PGX_LOG(AST_TRANSLATE, DEBUG,
                                "Mapped correlation paramid=%d via unified params to %s.%s (nullable=%d)", param_id,
                                resolved.table_name.c_str(), resolved.column_name.c_str(), resolved.nullable);
                    } else {
                        PGX_ERROR("Correlation arg is PARAM paramid=%d but not found in params map", param->paramid);
                        throw std::runtime_error("Correlation PARAM not found in nest_params");
                    }
                } else {
                    // Complex expression in correlation (e.g., t.col + 1)
                    PGX_ERROR("Correlation arg is complex expression (nodeTag=%d), not yet supported",
                              arg_expr ? nodeTag(arg_expr) : -1);
                    throw std::runtime_error("Complex correlation expressions not yet supported");
                }
            }
        }

        auto subquery_ctx = QueryCtxT::createChildContext(ctx);
        auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        for (const auto& [param_id, info] : correlation_mapping) {
            auto colRef = columnManager.createRef(info.table_scope, info.column_name);
            PostgreSQLTypeMapper mapper(*ctx.builder.getContext());
            auto mlir_type = mapper.map_postgre_sqltype(info.type_oid, info.typmod, info.nullable);

            mlir::Value correlation_value = ctx.builder.create<mlir::relalg::GetColumnOp>(
                ctx.builder.getUnknownLoc(),
                mlir_type,
                colRef,
                ctx.current_tuple
            );

            subquery_ctx.params[param_id] = pgx_lower::frontend::sql::ResolvedParam{
                .table_name = info.table_scope,
                .column_name = info.column_name,
                .type_oid = info.type_oid,
                .typmod = info.typmod,
                .nullable = info.nullable,
                .mlir_type = mlir_type,
                .cached_value = correlation_value
            };
        }
        auto [subquery_stream, subquery_result] = translate_subquery_plan(subquery_ctx, subquery_plan, &ctx.current_stmt);

        if (subquery_result.columns.empty()) {
            PGX_ERROR("Scalar subquery (plan_id=%d) returned no columns", subplan->plan_id);
            throw std::runtime_error("Scalar subquery must return exactly one column");
        }

        const auto& result_column = subquery_result.columns[0];
        auto column_ref = columnManager.createRef(result_column.table_name, result_column.column_name);

        mlir::Type result_type = result_column.mlir_type;
        if (!isa<mlir::db::NullableType>(result_type)) {
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

    case ANY_SUBLINK:
    case ALL_SUBLINK: {
        const bool is_all = (subplan->subLinkType == ALL_SUBLINK);
        PGX_LOG(AST_TRANSLATE, DEBUG, "%s: Translating %s pattern", is_all ? "ALL_SUBLINK" : "ANY_SUBLINK",
                is_all ? "x > ALL (subquery)" : "x IN (subquery)");

        auto translate_quantified = [&](bool negate_predicate) -> mlir::Value {
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
                PGX_ERROR("%s missing testexpr", negate_predicate ? "ALL_SUBLINK" : "ANY_SUBLINK");
                throw std::runtime_error("Quantified subquery requires testexpr");
            }

            const auto tuple_type = mlir::relalg::TupleType::get(ctx.builder.getContext());
            auto selection_op = ctx.builder.create<mlir::relalg::SelectionOp>(ctx.builder.getUnknownLoc(),
                                                                              subquery_stream);

            auto& pred_region = selection_op.getPredicate();
            auto& pred_block = pred_region.emplaceBlock();
            auto inner_tuple = pred_block.addArgument(tuple_type, ctx.builder.getUnknownLoc());

            mlir::OpBuilder pred_builder(&pred_block, pred_block.begin());

            auto inner_ctx = QueryCtxT(ctx.current_stmt, pred_builder, ctx.current_module, inner_tuple, mlir::Value());
            inner_ctx.params = ctx.params;
            inner_ctx.varno_resolution = ctx.varno_resolution;

            if (subplan->paramIds) {
                const int num_params = list_length(subplan->paramIds);
                PGX_LOG(AST_TRANSLATE, DEBUG, "%s: Mapping %d paramIds to subquery columns",
                        negate_predicate ? "ALL_SUBLINK" : "ANY_SUBLINK", num_params);

                for (int i = 0; i < num_params; ++i) {
                    const int param_id = lfirst_int(list_nth_cell(subplan->paramIds, i));

                    if (i < static_cast<int>(subquery_result.columns.size())) {
                        const auto& column_schema = subquery_result.columns[i];
                        inner_ctx.params[param_id] = pgx_lower::frontend::sql::ResolvedParam{
                            .table_name = column_schema.table_name,
                            .column_name = column_schema.column_name,
                            .type_oid = column_schema.type_oid,
                            .typmod = column_schema.typmod,
                            .nullable = column_schema.nullable,
                            .mlir_type = column_schema.mlir_type
                        };

                        PGX_LOG(AST_TRANSLATE, DEBUG, "  Mapped paramId=%d to column %s.%s (index %d)", param_id,
                                column_schema.table_name.c_str(), column_schema.column_name.c_str(), i);
                    } else {
                        PGX_ERROR("ParamId=%d index %d exceeds subquery columns (%zu)", param_id, i,
                                  subquery_result.columns.size());
                        throw std::runtime_error("ParamId index out of range");
                    }
                }
            }

            auto comparison = translate_expression(inner_ctx, reinterpret_cast<Expr*>(subplan->testexpr));
            if (!comparison) {
                PGX_ERROR("Failed to translate %s testexpr", negate_predicate ? "ALL_SUBLINK" : "ANY_SUBLINK");
                throw std::runtime_error("Failed to translate testexpr");
            }

            if (negate_predicate) {
                comparison = pred_builder.create<mlir::db::NotOp>(pred_builder.getUnknownLoc(), comparison.getType(),
                                                                  comparison);
            }

            pred_builder.create<mlir::relalg::ReturnOp>(pred_builder.getUnknownLoc(), mlir::ValueRange{comparison});

            auto exists_op = ctx.builder.create<mlir::relalg::ExistsOp>(
                ctx.builder.getUnknownLoc(), ctx.builder.getI1Type(), selection_op.getResult());

            if (negate_predicate) {
                auto final_not = ctx.builder.create<mlir::db::NotOp>(ctx.builder.getUnknownLoc(), exists_op.getType(),
                                                                     exists_op.getResult());
                PGX_LOG(AST_TRANSLATE, DEBUG, "ALL_SUBLINK: Created NOT EXISTS pattern");
                return final_not.getResult();
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "ANY_SUBLINK: Created EXISTS pattern");
            return exists_op.getResult();
        };

        return translate_quantified(is_all);
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
    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_subquery_plan: Starting subquery translation");
    auto subquery_ctx = QueryCtxT::createChildContext(parent_ctx, parent_ctx.builder, mlir::Value());
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

} // namespace postgresql_ast
