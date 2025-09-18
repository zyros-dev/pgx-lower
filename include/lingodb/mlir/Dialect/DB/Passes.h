#ifndef MLIR_DIALECT_DB_PASSES_H
#define MLIR_DIALECT_DB_PASSES_H
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace db {
std::unique_ptr<Pass> createEliminateNullsPass();
std::unique_ptr<Pass> createSimplifyToArithPass();
std::unique_ptr<Pass> createOptimizeRuntimeFunctionsPass();
std::unique_ptr<Pass> createInjectDecimalScalePass();
} // end namespace db
} // end namespace mlir
#endif // MLIR_DIALECT_DB_PASSES_H
