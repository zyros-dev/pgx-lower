#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/translation/translator_internals.h"

namespace postgresql_ast {

using TranslationContext = pgx_lower::frontend::sql::TranslationContext;
// Implementation of PostgreSQLASTTranslator public interface

PostgreSQLASTTranslator::PostgreSQLASTTranslator(mlir::MLIRContext& context)
: pImpl(std::make_unique<Impl>(context)) {}

PostgreSQLASTTranslator::~PostgreSQLASTTranslator() = default;

auto PostgreSQLASTTranslator::translate_query(PlannedStmt* plannedStmt) const -> std::unique_ptr<mlir::ModuleOp> {
    return pImpl->translate_query(plannedStmt);
}

auto PostgreSQLASTTranslator::Impl::translate_query(PlannedStmt* planned_stmt) -> std::unique_ptr<mlir::ModuleOp> {
    PGX_LOG(AST_TRANSLATE,
            IO,
            "translate_query IN: PostgreSQL PlannedStmt (cmd=%d, canSetTag=%d)",
            planned_stmt ? planned_stmt->commandType : -1,
            planned_stmt ? planned_stmt->canSetTag : false);

    if (!planned_stmt) {
        PGX_ERROR("PlannedStmt is null");
        return nullptr;
    }

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToStart(module.getBody());

    builder_ = &builder;
    current_planned_stmt_ = planned_stmt;

    TranslationContext context;
    context.currentStmt = planned_stmt;
    context.builder = &builder;

    auto queryFunc = create_query_function(builder, context);
    if (!queryFunc) {
        PGX_ERROR("Failed to create query function");
        builder_ = nullptr;
        current_planned_stmt_ = nullptr;
        return nullptr;
    }

    if (!generate_rel_alg_operations(queryFunc, planned_stmt, context)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        builder_ = nullptr;
        current_planned_stmt_ = nullptr;
        return nullptr;
    }

    builder_ = nullptr;
    current_planned_stmt_ = nullptr;

    auto result = std::make_unique<mlir::ModuleOp>(module);
    auto numOps = module.getBody()->getOperations().size();
    PGX_LOG(AST_TRANSLATE, IO, "translate_query OUT: RelAlg MLIR Module with %zu operations", numOps);

    return result;
}

auto PostgreSQLASTTranslator::Impl::generate_rel_alg_operations(mlir::func::FuncOp query_func,
                                                                const PlannedStmt* planned_stmt,
                                                                TranslationContext& context) -> bool {
    PGX_LOG(AST_TRANSLATE,
            IO,
            "generate_rel_alg_operations IN: PlannedStmt with planTree type %d",
            planned_stmt ? (planned_stmt->planTree ? planned_stmt->planTree->type : -1) : -1);

    if (!planned_stmt) {
        PGX_ERROR("PlannedStmt is null");
        return false;
    }

    Plan* planTree = planned_stmt->planTree;

    if (!validate_plan_tree(planTree)) {
        return false;
    }

    auto translatedOp = translate_plan_node(planTree, context);
    if (!translatedOp) {
        PGX_ERROR("Failed to translate plan node");
        return false;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Checking if translated operation has results");
    if (translatedOp->getNumResults() > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Operation has %d results", translatedOp->getNumResults());
        auto result = translatedOp->getResult(0);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Got result from translated operation");

        PGX_LOG(AST_TRANSLATE, DEBUG, "Checking result type");
        if (mlir::isa<mlir::relalg::TupleStreamType>(result.getType())) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Result is TupleStreamType, creating MaterializeOp");
            create_materialize_op(context, result);
        }
        else {
        }
    }
    else {
    }

    context.builder->create<mlir::func::ReturnOp>(context.builder->getUnknownLoc());

    PGX_LOG(AST_TRANSLATE, IO, "generate_rel_alg_operations OUT: RelAlg operations generated successfully");

    return true;
}

auto create_postgresql_ast_translator(mlir::MLIRContext& context) -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast