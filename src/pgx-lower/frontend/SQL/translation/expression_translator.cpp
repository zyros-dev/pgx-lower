#include "pgx-lower/frontend/SQL/translation/translation_context.h"
#include "pgx-lower/utility/logging.h"

// PostgreSQL headers (minimal, only what's absolutely needed)
extern "C" {
#include "postgres.h"
#include "nodes/primnodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"  // For Value type
#include "catalog/pg_type.h"
#include "catalog/pg_operator.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"  // For TextDatumGetCString

#define TextDatumGetCString(d) DatumGetCString(DirectFunctionCall1(textout, (d)))
}

// MLIR headers
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

// Dialect headers
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"

#include <string>
#include <vector>

namespace pgx_lower::ast::expression {

namespace {

// Helper to get table name from RTE
[[nodiscard]] auto get_table_name_from_rte(List* rtable, int varno) -> std::string {
    if (!rtable || varno <= 0) {
        return "unknown_table";
    }
    
    if (varno > list_length(rtable)) {
        PGX_WARNING("varno " + std::to_string(varno) + " exceeds rtable length");
        return "unknown_table";
    }
    
    auto* rte = static_cast<RangeTblEntry*>(list_nth(rtable, varno - 1));
    if (!rte) {
        return "unknown_table";
    }
    
    if (rte->rtekind == RTE_RELATION && rte->relid != 0) {
        char* relname = get_rel_name(rte->relid);
        if (relname) {
            std::string name(relname);
            pfree(relname);
            return name;
        }
    }
    
    if (rte->eref && rte->eref->aliasname) {
        return std::string(rte->eref->aliasname);
    }
    
    return "unknown_table";
}

// Helper to get column name from schema
[[nodiscard]] auto get_column_name_from_schema(List* rtable, int varno, int varattno) -> std::string {
    if (!rtable || varno <= 0 || varattno <= 0) {
        return "unknown_column";
    }
    
    if (varno > list_length(rtable)) {
        return "column_" + std::to_string(varattno);
    }
    
    auto* rte = static_cast<RangeTblEntry*>(list_nth(rtable, varno - 1));
    if (!rte) {
        return "column_" + std::to_string(varattno);
    }
    
    if (rte->rtekind == RTE_RELATION && rte->relid != 0) {
        char* attname = get_attname(rte->relid, varattno, false);
        if (attname) {
            std::string name(attname);
            pfree(attname);
            return name;
        }
    }
    
    if (rte->eref && rte->eref->colnames && varattno <= list_length(rte->eref->colnames)) {
        auto* colname = static_cast<struct Value*>(list_nth(rte->eref->colnames, varattno - 1));
        if (colname) {
            return std::string(strVal(colname));
        }
    }
    
    return "column_" + std::to_string(varattno);
}

// Map PostgreSQL type OID to MLIR type
[[nodiscard]] auto map_postgresql_type(Oid type_oid, int32_t typmod, ::mlir::MLIRContext& context) -> ::mlir::Type {
    switch (type_oid) {
    case BOOLOID:
        return ::mlir::IntegerType::get(&context, 1);
    case INT2OID:
        return ::mlir::IntegerType::get(&context, 16);
    case INT4OID:
        return ::mlir::IntegerType::get(&context, 32);
    case INT8OID:
        return ::mlir::IntegerType::get(&context, 64);
    case FLOAT4OID:
        return ::mlir::Float32Type::get(&context);
    case FLOAT8OID:
        return ::mlir::Float64Type::get(&context);
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID:
        return ::mlir::db::StringType::get(&context);
    default:
        PGX_WARNING("Unsupported PostgreSQL type OID: " + std::to_string(type_oid) + ", using i32");
        return ::mlir::IntegerType::get(&context, 32);
    }
}

// Forward declarations for mutual recursion
[[nodiscard]] auto translate_expression_impl(Expr* expr, TranslationContext& ctx) -> ::mlir::Value;

// Translate variable reference
[[nodiscard]] auto translate_var(Var* var, TranslationContext& ctx) -> ::mlir::Value {
    if (!var || !ctx.builder) {
        PGX_ERROR("Invalid Var parameters");
        return nullptr;
    }
    
    // If we're in a tuple context (e.g., inside MapOp or SelectionOp)
    if (ctx.current_block && !ctx.current_block->getArguments().empty()) {
        auto tuple_arg = ctx.current_block->getArgument(0);
        
        std::string table_name = get_table_name_from_rte(ctx.rtable, var->varno);
        std::string col_name = get_column_name_from_schema(ctx.rtable, var->varno, var->varattno);
        
        auto* dialect = ctx.mlir_context->getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        if (!dialect) {
            PGX_ERROR("RelAlg dialect not registered");
            return nullptr;
        }
        
        auto& column_manager = dialect->getColumnManager();
        auto mlir_type = map_postgresql_type(var->vartype, var->vartypmod, *ctx.mlir_context);
        
        auto col_ref = column_manager.createRef(table_name, col_name);
        col_ref.getColumn().type = mlir_type;
        
        auto get_col_op = ctx.builder->create<mlir::relalg::GetColumnOp>(
            ctx.builder->getUnknownLoc(), mlir_type, col_ref, tuple_arg);
        
        return get_col_op.getRes();
    }
    
    PGX_WARNING("No tuple context for Var translation");
    return ctx.builder->create<mlir::arith::ConstantIntOp>(
        ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
}

// Translate constant value
[[nodiscard]] auto translate_const(Const* const_node, TranslationContext& ctx) -> ::mlir::Value {
    if (!const_node || !ctx.builder) {
        PGX_ERROR("Invalid Const parameters");
        return nullptr;
    }
    
    if (const_node->constisnull) {
        auto null_type = mlir::db::NullableType::get(
            ctx.mlir_context, mlir::IntegerType::get(ctx.mlir_context, 32));
        return ctx.builder->create<mlir::db::NullOp>(ctx.builder->getUnknownLoc(), null_type);
    }
    
    auto mlir_type = map_postgresql_type(const_node->consttype, const_node->consttypmod, *ctx.mlir_context);
    
    switch (const_node->consttype) {
    case INT4OID: {
        int32_t val = static_cast<int32_t>(const_node->constvalue);
        return ctx.builder->create<mlir::db::ConstantOp>(
            ctx.builder->getUnknownLoc(), ctx.builder->getI32Type(), ctx.builder->getI32IntegerAttr(val));
    }
    case INT8OID: {
        int64_t val = static_cast<int64_t>(const_node->constvalue);
        return ctx.builder->create<mlir::db::ConstantOp>(
            ctx.builder->getUnknownLoc(), ctx.builder->getI64Type(), ctx.builder->getI64IntegerAttr(val));
    }
    case BOOLOID: {
        bool val = static_cast<bool>(const_node->constvalue);
        return ctx.builder->create<mlir::db::ConstantOp>(
            ctx.builder->getUnknownLoc(), ctx.builder->getI1Type(), ctx.builder->getBoolAttr(val));
    }
    case FLOAT4OID: {
        float val = *reinterpret_cast<float*>(&const_node->constvalue);
        return ctx.builder->create<mlir::arith::ConstantFloatOp>(
            ctx.builder->getUnknownLoc(), llvm::APFloat(val), ctx.builder->getF32Type());
    }
    case FLOAT8OID: {
        double val = *reinterpret_cast<double*>(&const_node->constvalue);
        return ctx.builder->create<mlir::arith::ConstantFloatOp>(
            ctx.builder->getUnknownLoc(), llvm::APFloat(val), ctx.builder->getF64Type());
    }
    case TEXTOID:
    case VARCHAROID: {
        char* str = TextDatumGetCString(const_node->constvalue);
        auto str_attr = ctx.builder->getStringAttr(str);
        pfree(str);
        return ctx.builder->create<mlir::db::ConstantOp>(
            ctx.builder->getUnknownLoc(), mlir::db::StringType::get(ctx.mlir_context), str_attr);
    }
    default:
        PGX_WARNING("Unsupported constant type: " + std::to_string(const_node->consttype));
        return ctx.builder->create<mlir::arith::ConstantIntOp>(
            ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
    }
}

// Translate arithmetic operations
[[nodiscard]] auto translate_arithmetic_op(Oid op_oid, ::mlir::Value lhs, ::mlir::Value rhs, TranslationContext& ctx) -> ::mlir::Value {
    if (!lhs || !rhs) {
        PGX_ERROR("Invalid operands for arithmetic operation");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    
    switch (op_oid) {
    case 551:  // int4pl (addition)
    case 550:  // int2pl
    case 463:  // int8pl
        return ctx.builder->create<mlir::db::AddOp>(loc, lhs, rhs);
        
    case 555:  // int4mi (subtraction)
    case 554:  // int2mi
    case 464:  // int8mi
        return ctx.builder->create<mlir::db::SubOp>(loc, lhs, rhs);
        
    case 514:  // int4mul (multiplication)
    case 524:  // int2mul
    case 465:  // int8mul
        return ctx.builder->create<mlir::db::MulOp>(loc, lhs, rhs);
        
    case 528:  // int4div (division)
    case 527:  // int2div
    case 466:  // int8div
        return ctx.builder->create<mlir::db::DivOp>(loc, lhs, rhs);
        
    case 530:  // int4mod (modulo)
    case 529:  // int2mod
    case 439:  // int8mod
        return ctx.builder->create<mlir::arith::RemSIOp>(loc, lhs, rhs);
        
    default:
        PGX_WARNING("Unsupported arithmetic operator OID: " + std::to_string(op_oid));
        return nullptr;
    }
}

// Translate comparison operations
[[nodiscard]] auto translate_comparison_op(Oid op_oid, ::mlir::Value lhs, ::mlir::Value rhs, TranslationContext& ctx) -> ::mlir::Value {
    if (!lhs || !rhs) {
        PGX_ERROR("Invalid operands for comparison operation");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    
    switch (op_oid) {
    case 96:   // int4eq (=)
    case 94:   // int2eq
    case 410:  // int8eq
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, lhs, rhs);
        
    case 518:  // int4ne (!=)
    case 519:  // int2ne
    case 411:  // int8ne
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::neq, lhs, rhs);
        
    case 97:   // int4lt (<)
    case 95:   // int2lt
    case 412:  // int8lt
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, lhs, rhs);
        
    case 521:  // int4gt (>)
    case 520:  // int2gt
    case 413:  // int8gt
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gt, lhs, rhs);
        
    case 523:  // int4le (<=)
    case 522:  // int2le
    case 414:  // int8le
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lte, lhs, rhs);
        
    case 525:  // int4ge (>=)
    case 524:  // int2ge
    case 415:  // int8ge
        return ctx.builder->create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gte, lhs, rhs);
        
    default:
        PGX_WARNING("Unsupported comparison operator OID: " + std::to_string(op_oid));
        return nullptr;
    }
}

// Extract operands from OpExpr
[[nodiscard]] auto extract_op_expr_operands(OpExpr* op_expr, TranslationContext& ctx) -> std::pair<::mlir::Value, ::mlir::Value> {
    if (!op_expr || !op_expr->args) {
        PGX_ERROR("Invalid OpExpr or missing arguments");
        return {nullptr, nullptr};
    }
    
    if (list_length(op_expr->args) != 2) {
        PGX_WARNING("OpExpr with " + std::to_string(list_length(op_expr->args)) + " args, expected 2");
        return {nullptr, nullptr};
    }
    
    auto* left_expr = static_cast<Expr*>(linitial(op_expr->args));
    auto* right_expr = static_cast<Expr*>(lsecond(op_expr->args));
    
    auto lhs = translate_expression_impl(left_expr, ctx);
    auto rhs = translate_expression_impl(right_expr, ctx);
    
    return {lhs, rhs};
}

// Translate operator expression
[[nodiscard]] auto translate_op_expr(OpExpr* op_expr, TranslationContext& ctx) -> ::mlir::Value {
    if (!op_expr) {
        PGX_ERROR("OpExpr is null");
        return nullptr;
    }
    
    auto [lhs, rhs] = extract_op_expr_operands(op_expr, ctx);
    if (!lhs || !rhs) {
        return nullptr;
    }
    
    // Try arithmetic operators first
    if (auto result = translate_arithmetic_op(op_expr->opno, lhs, rhs, ctx)) {
        return result;
    }
    
    // Try comparison operators
    if (auto result = translate_comparison_op(op_expr->opno, lhs, rhs, ctx)) {
        return result;
    }
    
    PGX_WARNING("Unsupported operator OID: " + std::to_string(op_expr->opno));
    return ctx.builder->create<mlir::arith::ConstantIntOp>(
        ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
}

// Translate boolean expression
[[nodiscard]] auto translate_bool_expr(BoolExpr* bool_expr, TranslationContext& ctx) -> ::mlir::Value {
    if (!bool_expr || !bool_expr->args) {
        PGX_ERROR("Invalid BoolExpr");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    std::vector<::mlir::Value> operands;
    
    // Translate all operands
    ListCell* lc;
    foreach(lc, bool_expr->args) {
        auto* expr = static_cast<Expr*>(lfirst(lc));
        if (auto val = translate_expression_impl(expr, ctx)) {
            operands.push_back(val);
        }
    }
    
    if (operands.empty()) {
        PGX_ERROR("No valid operands for BoolExpr");
        return nullptr;
    }
    
    switch (bool_expr->boolop) {
    case AND_EXPR: {
        auto result = operands[0];
        for (size_t i = 1; i < operands.size(); ++i) {
            result = ctx.builder->create<mlir::db::AndOp>(loc, ctx.builder->getI1Type(), mlir::ValueRange{result, operands[i]});
        }
        return result;
    }
    case OR_EXPR: {
        auto result = operands[0];
        for (size_t i = 1; i < operands.size(); ++i) {
            result = ctx.builder->create<mlir::db::OrOp>(loc, ctx.builder->getI1Type(), mlir::ValueRange{result, operands[i]});
        }
        return result;
    }
    case NOT_EXPR: {
        if (operands.size() != 1) {
            PGX_ERROR("NOT_EXPR expects exactly one operand");
            return nullptr;
        }
        return ctx.builder->create<mlir::db::NotOp>(loc, operands[0]);
    }
    default:
        PGX_ERROR("Unsupported boolean operation: " + std::to_string(bool_expr->boolop));
        return nullptr;
    }
}

// Translate function expression
[[nodiscard]] auto translate_func_expr(FuncExpr* func_expr, TranslationContext& ctx) -> ::mlir::Value {
    if (!func_expr) {
        PGX_ERROR("FuncExpr is null");
        return nullptr;
    }
    
    // For now, we'll create a placeholder for most functions
    // This will be expanded as we add support for more PostgreSQL functions
    
    std::vector<::mlir::Value> args;
    if (func_expr->args) {
        ListCell* lc;
        foreach(lc, func_expr->args) {
            auto* expr = static_cast<Expr*>(lfirst(lc));
            if (auto val = translate_expression_impl(expr, ctx)) {
                args.push_back(val);
            }
        }
    }
    
    // Handle some common functions
    switch (func_expr->funcid) {
    // Add specific function handling here as needed
    default:
        PGX_WARNING("Unsupported function OID: " + std::to_string(func_expr->funcid));
        return ctx.builder->create<mlir::arith::ConstantIntOp>(
            ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
    }
}

// Translate aggregate reference
[[nodiscard]] auto translate_aggref(Aggref* aggref, TranslationContext& ctx) -> ::mlir::Value {
    if (!aggref) {
        PGX_ERROR("Aggref is null");
        return nullptr;
    }
    
    // Aggregates need special handling in the context of an Agg plan node
    // For now, return a placeholder
    PGX_WARNING("Aggregate functions not yet fully implemented");
    return ctx.builder->create<mlir::arith::ConstantIntOp>(
        ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
}

// Translate null test
[[nodiscard]] auto translate_null_test(NullTest* null_test, TranslationContext& ctx) -> ::mlir::Value {
    if (!null_test || !null_test->arg) {
        PGX_ERROR("Invalid NullTest");
        return nullptr;
    }
    
    auto arg = translate_expression_impl(static_cast<Expr*>(null_test->arg), ctx);
    if (!arg) {
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    auto is_null = ctx.builder->create<mlir::db::IsNullOp>(loc, arg);
    
    if (null_test->nulltesttype == IS_NOT_NULL) {
        return ctx.builder->create<mlir::db::NotOp>(loc, is_null);
    }
    
    return is_null;
}

// Translate coalesce expression
[[nodiscard]] auto translate_coalesce_expr(CoalesceExpr* coalesce_expr, TranslationContext& ctx) -> ::mlir::Value {
    if (!coalesce_expr || !coalesce_expr->args) {
        PGX_ERROR("Invalid CoalesceExpr");
        return nullptr;
    }
    
    // COALESCE returns the first non-null argument
    // For now, return the first argument as a placeholder
    auto* first_expr = static_cast<Expr*>(linitial(coalesce_expr->args));
    return translate_expression_impl(first_expr, ctx);
}

// Main expression translation dispatcher
[[nodiscard]] auto translate_expression_impl(Expr* expr, TranslationContext& ctx) -> ::mlir::Value {
    if (!expr) {
        PGX_ERROR("Expression is null");
        return nullptr;
    }
    
    // Safety check for valid pointer
    if (reinterpret_cast<uintptr_t>(expr) < 0x1000) {
        PGX_ERROR("Invalid expression pointer");
        return nullptr;
    }
    
    switch (expr->type) {
    case T_Var:
        return translate_var(reinterpret_cast<Var*>(expr), ctx);
    case T_Const:
        return translate_const(reinterpret_cast<Const*>(expr), ctx);
    case T_OpExpr:
        return translate_op_expr(reinterpret_cast<OpExpr*>(expr), ctx);
    case T_FuncExpr:
        return translate_func_expr(reinterpret_cast<FuncExpr*>(expr), ctx);
    case T_BoolExpr:
        return translate_bool_expr(reinterpret_cast<BoolExpr*>(expr), ctx);
    case T_Aggref:
        return translate_aggref(reinterpret_cast<Aggref*>(expr), ctx);
    case T_NullTest:
        return translate_null_test(reinterpret_cast<NullTest*>(expr), ctx);
    case T_CoalesceExpr:
        return translate_coalesce_expr(reinterpret_cast<CoalesceExpr*>(expr), ctx);
    default:
        PGX_WARNING("Unsupported expression type: " + std::to_string(expr->type));
        return ctx.builder->create<mlir::arith::ConstantIntOp>(
            ctx.builder->getUnknownLoc(), 0, ctx.builder->getI32Type());
    }
}

} // anonymous namespace

// Public interface
[[nodiscard]] auto translate(Expr* expr, TranslationContext& ctx) -> ::mlir::Value {
    return translate_expression_impl(expr, ctx);
}

} // namespace pgx_lower::ast::expression