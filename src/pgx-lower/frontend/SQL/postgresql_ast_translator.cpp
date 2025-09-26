#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/translation/translator_internals.h"

namespace postgresql_ast {

using QueryCtxT = pgx_lower::frontend::sql::TranslationContext;
using TranslationResult = pgx_lower::frontend::sql::TranslationResult;

PostgreSQLASTTranslator::PostgreSQLASTTranslator(mlir::MLIRContext& context)
: p_impl_(std::make_unique<Impl>(context)) {}

PostgreSQLASTTranslator::~PostgreSQLASTTranslator() = default;

auto PostgreSQLASTTranslator::translate_query(PlannedStmt* planned_stmt) const -> std::unique_ptr<mlir::ModuleOp> {
    PGX_IO(AST_TRANSLATE);
    return p_impl_->translate_query(planned_stmt);
}

auto PostgreSQLASTTranslator::Impl::translate_query(const PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp> {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, IO, "translate_query IN: PostgreSQL PlannedStmt (cmd=%d, canSetTag=%d)",
            planned_stmt ? planned_stmt->commandType : -1, planned_stmt ? planned_stmt->canSetTag : false);

    if (!planned_stmt) {
        PGX_ERROR("PlannedStmt is null");
        return nullptr;
    }

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    auto builder = mlir::OpBuilder(&context_);
    builder.setInsertionPointToStart(module.getBody());
    auto context = QueryCtxT{*planned_stmt, builder, module, nullptr, mlir::Value()};

    auto query_func = create_query_function(builder);
    if (!query_func) {
        PGX_ERROR("Failed to create query function");
        return nullptr;
    }

    if (!generate_rel_alg_operations(planned_stmt, context)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        return nullptr;
    }

    auto result = std::make_unique<mlir::ModuleOp>(module);
    const auto num_ops = module.getBody()->getOperations().size();
    PGX_LOG(AST_TRANSLATE, IO, "translate_query OUT: RelAlg MLIR Module with %zu operations", num_ops);

    return result;
}

auto PostgreSQLASTTranslator::Impl::generate_rel_alg_operations(const PlannedStmt* planned_stmt, QueryCtxT& context)
    -> bool {
    PGX_IO(AST_TRANSLATE);
    PGX_LOG(AST_TRANSLATE, IO, "generate_rel_alg_operations IN: PlannedStmt with planTree type %d",
            planned_stmt ? (planned_stmt->planTree ? planned_stmt->planTree->type : -1) : -1);
    assert(planned_stmt);
    auto* plan_tree = planned_stmt->planTree;

    const auto translationResult = translate_plan_node(context, plan_tree);
    if (!translationResult.op) {
        PGX_ERROR("Failed to translate plan node");
        return false;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Checking if translated operation has results");
    mlir::Value returnValue;
    if (translationResult.op->getNumResults() > 0) {
        const auto result = translationResult.op->getResult(0);
        if (mlir::isa<mlir::relalg::TupleStreamType>(result.getType())) {
            returnValue = create_materialize_op(context, result, translationResult);
        }
    }

    if (returnValue) {
        context.builder.create<mlir::func::ReturnOp>(context.builder.getUnknownLoc(), returnValue);
    } else {
        context.builder.create<mlir::func::ReturnOp>(context.builder.getUnknownLoc());
    }
    PGX_LOG(AST_TRANSLATE, IO, "generate_rel_alg_operations OUT: RelAlg operations generated successfully");

    return true;
}

auto create_postgresql_ast_translator(mlir::MLIRContext& context) -> std::unique_ptr<PostgreSQLASTTranslator> {
    PGX_IO(AST_TRANSLATE);
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast