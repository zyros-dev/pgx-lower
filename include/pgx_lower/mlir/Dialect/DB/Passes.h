#ifndef PGX_LOWER_MLIR_DIALECT_DB_PASSES_H
#define PGX_LOWER_MLIR_DIALECT_DB_PASSES_H
#include "mlir/Pass/Pass.h"

namespace pgx {
namespace mlir {
namespace db {
std::unique_ptr<::mlir::Pass> createEliminateNullsPass();
std::unique_ptr<::mlir::Pass> createSimplifyToArithPass();
std::unique_ptr<::mlir::Pass> createOptimizeRuntimeFunctionsPass();
} // end namespace db
} // end namespace mlir
} // end namespace pgx
#endif // PGX_LOWER_MLIR_DIALECT_DB_PASSES_H