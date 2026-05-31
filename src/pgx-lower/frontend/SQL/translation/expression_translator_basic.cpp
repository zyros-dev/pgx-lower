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

auto PostgreSQLASTTranslator::Impl::translate_expression(const QueryCtxT& ctx, Expr* expr) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, DEBUG, "Parsing %d", expr->type);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: expr->type=%d", expr->type);

    if (!expr) {
        PGX_ERROR("Expression is null");
        throw std::runtime_error("Expression cannot be null");
    }

    switch (expr->type) {
    case T_Var:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_Var");
        return translate_var(ctx, reinterpret_cast<Var*>(expr));
    case T_Const:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_Const");
        return translate_const(ctx, reinterpret_cast<Const*>(expr));
    case T_OpExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_OpExpr");
        return translate_op_expr(ctx, reinterpret_cast<OpExpr*>(expr));
    case T_FuncExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_FuncExpr");
        return translate_func_expr(ctx, reinterpret_cast<FuncExpr*>(expr));
    case T_BoolExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_BoolExpr");
        return translate_bool_expr(ctx, reinterpret_cast<BoolExpr*>(expr));
    case T_Aggref:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_Aggref");
        return translate_aggref(ctx, reinterpret_cast<Aggref*>(expr));
    case T_NullTest:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_NullTest");
        return translate_null_test(ctx, reinterpret_cast<NullTest*>(expr));
    case T_CoalesceExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_CoalesceExpr");
        return translate_coalesce_expr(ctx, reinterpret_cast<CoalesceExpr*>(expr));
    case T_ScalarArrayOpExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_ScalarArrayOpExpr");
        return translate_scalar_array_op_expr(ctx, reinterpret_cast<ScalarArrayOpExpr*>(expr));
    case T_CaseExpr:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_CaseExpr");
        return translate_case_expr(ctx, reinterpret_cast<CaseExpr*>(expr));
    case T_CoerceViaIO:
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_CoerceViaIO");
        return translate_coerce_via_io(ctx, expr);
    case T_RelabelType: {
        const auto* relabel = reinterpret_cast<RelabelType*>(expr);
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_expression: CASE=T_RelabelType");
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unwrapping T_RelabelType to translate underlying expression");
        return translate_expression(ctx, relabel->arg);
    }
    case T_SubPlan: return translate_subplan(ctx, reinterpret_cast<SubPlan*>(expr));
    case T_Param: return translate_param(ctx, reinterpret_cast<Param*>(expr));
    default: {
        PGX_ERROR("Unsupported expression type: %d", expr->type);
        throw std::runtime_error("Unsupported expression type - read the logs");
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translate_var(const QueryCtxT& ctx, const Var* var) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!var) {
        PGX_ERROR("Invalid Var: var is null");
        throw std::runtime_error("Invalid Var parameters");
    }

    if (!ctx.current_tuple) {
        PGX_ERROR("Invalid Var parameters: var=%p, builder=%p, tuple=%p", var, ctx.builder,
                  ctx.current_tuple.getAsOpaquePointer());
        throw std::runtime_error("Invalid Var parameters: no tuple");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_var: varno=%d, varattno=%d", var->varno, var->varattno);

    std::string tableName, colName;
    bool nullable;
    bool resolved_from_mapping = false;

    std::optional<int> varnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varnosyn) : std::nullopt;
    std::optional<int> varattnosyn_opt = IS_SPECIAL_VARNO(var->varno) ? std::optional<int>(var->varattnosyn)
                                                                      : std::nullopt;

    if (auto resolved = ctx.resolve_var(var->varno, var->varattno, varnosyn_opt, varattnosyn_opt)) {
        tableName = resolved->table_name;
        colName = resolved->column_name;
        nullable = resolved->nullable;
        resolved_from_mapping = true;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using varno_resolution for varno=%d, varattno=%d -> (%s, %s, nullable=%d)",
                var->varno, var->varattno, tableName.c_str(), colName.c_str(), nullable);
    }

    if (!resolved_from_mapping && var->varno == OUTER_VAR) {
        auto& result_to_use = ctx.outer_result ? ctx.outer_result.value()
                                               : throw std::runtime_error("OUTER_VAR without outer_result");

        if (var->varattno <= 0 || var->varattno > static_cast<int>(result_to_use.get().columns.size())) {
            PGX_ERROR("OUTER_VAR varattno=%d out of range (result has %zu columns)", var->varattno,
                      result_to_use.get().columns.size());
            throw std::runtime_error("OUTER_VAR reference out of range");
        }
        const auto& col = result_to_use.get().columns[var->varattno - 1];
        tableName = col.table_name;
        colName = col.column_name;
        nullable = col.nullable;
        resolved_from_mapping = true;
        PGX_LOG(AST_TRANSLATE, DEBUG, "OUTER_VAR varattno=%d resolved to %s.%s (nullable=%d) from outer_result",
                var->varattno, tableName.c_str(), colName.c_str(), nullable);
        PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_var: BRANCH=OUTER_VAR, tableName='%s', colName='%s'",
                tableName.c_str(), colName.c_str());
    }

    if (!resolved_from_mapping && (var->varno == INNER_VAR || var->varno == INDEX_VAR)) {
        PGX_ERROR("INNER_VAR/INDEX_VAR varno=%d varattno=%d not found in varno_resolution - join translation bug",
                  var->varno, var->varattno);
        throw std::runtime_error("INNER_VAR/INDEX_VAR requires varno_resolution mapping");
    }

    if (!resolved_from_mapping) {
        // Final fallback: use PostgreSQL catalog
        // For synthetic varnos, need to get concrete varno for schema lookup
        int schema_varno = IS_SPECIAL_VARNO(var->varno) ? var->varnosyn : var->varno;
        tableName = get_table_alias_from_rte(&ctx.current_stmt, schema_varno);
        colName = get_column_name_from_schema(&ctx.current_stmt, schema_varno, var->varattno);
        nullable = is_column_nullable(&ctx.current_stmt, schema_varno, var->varattno);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Fallback to schema lookup: varno=%d -> %s.%s", schema_varno, tableName.c_str(),
                colName.c_str());
    }

    auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        PGX_ERROR("RelAlg dialect not registered");
        throw std::runtime_error("Check logs");
    }

    auto& columnManager = dialect->getColumnManager();

    const auto type_mapper = PostgreSQLTypeMapper(context_);
    auto mlirType = type_mapper.map_postgre_sqltype(var->vartype, var->vartypmod, nullable);

    PGX_LOG(AST_TRANSLATE, DEBUG, "[SCOPE_DEBUG] translate_var: Creating GetColumnOp with scope='%s', column='%s'",
            tableName.c_str(), colName.c_str());

    auto colRef = columnManager.createRef(tableName, colName);

    // TODO: it's a bit goofy that we even need this safety check here
    if (!colRef.getColumn().type) {
        colRef.getColumn().type = mlirType;
    } else {
        mlirType = colRef.getColumn().type;
    }

    auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), mlirType, colRef,
                                                                  ctx.current_tuple);

    return getColOp.getRes();
}

auto PostgreSQLASTTranslator::Impl::translate_const(const QueryCtxT& ctx, Const* const_node) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    return postgresql_ast::translate_const(const_node, ctx.builder, context_);
}

auto PostgreSQLASTTranslator::Impl::translate_aggref(const QueryCtxT& ctx, const Aggref* aggref) const -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!aggref) {
        PGX_ERROR("Invalid Aggref parameters");
        throw std::runtime_error("Invalid Aggref parameters");
    }

    char* rawFuncName = get_func_name(aggref->aggfnoid);
    if (rawFuncName == nullptr) {
        PGX_ERROR("Unknown aggregate function OID: %u", aggref->aggfnoid);
        throw std::runtime_error("Unknown aggregate function OID");
    }
    const std::string funcName(rawFuncName);
    pfree(rawFuncName);

    PGX_LOG(AST_TRANSLATE, DEBUG, "translate_aggref: Looking for Aggref with function %s (OID %u, aggno=%d, aggtype=%d)",
            funcName.c_str(), aggref->aggfnoid, aggref->aggno, aggref->aggtype);

    std::string scopeName, columnName;

    bool found = false;
    if (auto resolved = ctx.resolve_var(-2, aggref->aggno)) {
        scopeName = resolved->table_name;
        columnName = resolved->column_name;
        found = true;
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using TranslationResult mapping for aggregate aggno=%d -> (%s, %s)",
                aggref->aggno, scopeName.c_str(), columnName.c_str());
    }
    if (!found) {
        PGX_ERROR("No mapping found for aggregate aggno=%d", aggref->aggno);
        throw std::runtime_error("Aggregate reference not found in column mappings");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Found Aggref mapping: aggno=%d -> scope=%s, column=%s", aggref->aggno,
            scopeName.c_str(), columnName.c_str());

    auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        PGX_ERROR("RelAlg dialect not registered");
        throw std::runtime_error("RelAlg dialect not registered");
    }
    auto& columnManager = dialect->getColumnManager();

    // Create column reference using the constructed scope and column name
    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating Aggref to GetColumnOp: scope=%s, column=%s", scopeName.c_str(),
            columnName.c_str());
    auto colRef = columnManager.createRef(scopeName, columnName);

    // Get the actual type from the column that was created during aggregation
    auto resultType = colRef.getColumn().type;
    if (!resultType) {
        std::string errorMsg = "Aggregate column type not found in column manager for scope='" + scopeName
                               + "', column='" + columnName + "'";
        PGX_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    auto getColOp = ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), resultType, colRef,
                                                                  ctx.current_tuple);
    return getColOp.getRes();
}

auto PostgreSQLASTTranslator::Impl::translate_param(const QueryCtxT& ctx, const Param* param) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);

    if (!param) {
        PGX_ERROR("Invalid Param node");
        throw std::runtime_error("Invalid Param node");
    }

    if (param->paramkind != PARAM_EXEC) {
        PGX_ERROR("Only PARAM_EXEC parameters are supported (got paramkind=%d)", param->paramkind);
        throw std::runtime_error("Unsupported param kind");
    }

    const auto it = ctx.params.find(param->paramid);
    if (it == ctx.params.end()) {
        PGX_ERROR("Unknown paramid=%d (not registered in params map)", param->paramid);
        throw std::runtime_error("Unknown paramid");
    }

    const auto& resolved = it->second;
    PGX_LOG(AST_TRANSLATE, DEBUG, "Resolved paramid=%d -> %s.%s", param->paramid, resolved.table_name.c_str(),
            resolved.column_name.c_str());

    if (resolved.cached_value) {
        auto cachedValue = *resolved.cached_value;
        if (!mlir::isa<mlir::relalg::TupleStreamType>(cachedValue.getType())) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Using cached correlation scalar for paramid=%d", param->paramid);
            return cachedValue;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "Using cached InitPlan tuplestream for paramid=%d", param->paramid);
        auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        auto column_ref = columnManager.createRef(resolved.table_name, resolved.column_name);
        const mlir::Value scalar_value = ctx.builder.create<mlir::relalg::GetScalarOp>(
            ctx.builder.getUnknownLoc(), resolved.mlir_type, column_ref, cachedValue);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Created GetScalarOp for InitPlan paramid=%d from %s.%s", param->paramid,
                resolved.table_name.c_str(), resolved.column_name.c_str());
        return scalar_value;
    }

    auto& columnManager = ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    auto colRef = columnManager.createRef(resolved.table_name, resolved.column_name);

    if (!colRef.getColumn().type) {
        colRef.getColumn().type = resolved.mlir_type;
    }

    return ctx.builder.create<mlir::relalg::GetColumnOp>(ctx.builder.getUnknownLoc(), resolved.mlir_type, colRef,
                                                         ctx.current_tuple);
}

} // namespace postgresql_ast