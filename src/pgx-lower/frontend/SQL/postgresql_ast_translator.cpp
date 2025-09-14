#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/translation/translator_internals.h"

namespace postgresql_ast {

using QueryCtxT = pgx_lower::frontend::sql::TranslationContext;

PostgreSQLASTTranslator::PostgreSQLASTTranslator(mlir::MLIRContext& context)
: p_impl_(std::make_unique<Impl>(context)) {}

PostgreSQLASTTranslator::~PostgreSQLASTTranslator() = default;

auto PostgreSQLASTTranslator::translate_query(PlannedStmt* planned_stmt) const -> std::unique_ptr<mlir::ModuleOp> {
    return p_impl_->translate_query(planned_stmt);
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

    auto context = QueryCtxT{};

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    auto builder = mlir::OpBuilder(&context_);
    builder.setInsertionPointToStart(module.getBody());

    context.current_stmt = planned_stmt;
    context.builder = &builder;
    context.current_module = &module;

    auto query_func = create_query_function(builder, context);
    if (!query_func) {
        PGX_ERROR("Failed to create query function");
        return nullptr;
    }

    if (!generate_rel_alg_operations(query_func, planned_stmt, context)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        return nullptr;
    }

    auto result = std::make_unique<mlir::ModuleOp>(module);
    const auto num_ops = module.getBody()->getOperations().size();
    PGX_LOG(AST_TRANSLATE, IO, "translate_query OUT: RelAlg MLIR Module with %zu operations", num_ops);

    return result;
}

auto PostgreSQLASTTranslator::Impl::generate_rel_alg_operations(mlir::func::FuncOp query_func,
                                                                const PlannedStmt* planned_stmt,
                                                                QueryCtxT& context) -> bool {
    PGX_LOG(AST_TRANSLATE,
            IO,
            "generate_rel_alg_operations IN: PlannedStmt with planTree type %d",
            planned_stmt ? (planned_stmt->planTree ? planned_stmt->planTree->type : -1) : -1);
    assert(planned_stmt);
    auto* plan_tree = planned_stmt->planTree;

    if (!validate_plan_tree(plan_tree)) {
        return false;
    }

    const auto translated_op = translate_plan_node(context, plan_tree);
    if (!translated_op) {
        PGX_ERROR("Failed to translate plan node");
        return false;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Checking if translated operation has results");
    if (translated_op->getNumResults() > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Operation has %d results", translated_op->getNumResults());
        const auto result = translated_op->getResult(0);
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