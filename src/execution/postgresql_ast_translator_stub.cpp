// Include PostgreSQL headers first with proper C linkage
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/nodeFuncs.h"  // For exprType
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
}

// Undefine PostgreSQL macros that conflict with LLVM
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "execution/postgresql_ast_translator.h"
#include "execution/logging.h"
#include "runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <memory>
#include <stdexcept>

// Clean slate refactor: Minimal stub implementation
// Will be rebuilt incrementally using LingoDB 2022 architecture

namespace postgresql_ast {

PostgreSQLASTTranslator::PostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    : context_(context), logger_(logger), builder_(nullptr), currentModule_(nullptr), 
      currentTupleHandle_(nullptr), currentPlannedStmt_(nullptr), contextNeedsRecreation_(false) {
    PGX_DEBUG("PostgreSQLASTTranslator stub implementation initialized");
}

auto PostgreSQLASTTranslator::translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<mlir::ModuleOp> {
    PGX_ERROR("PostgreSQLASTTranslator stub: MLIR dialects not yet implemented in clean slate refactor");
    ereport(ERROR, 
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("PostgreSQL AST translation not available - clean slate refactor in progress")));
    return nullptr;
}

auto createPostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context, logger);
}

} // namespace postgresql_ast