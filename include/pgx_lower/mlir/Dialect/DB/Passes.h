#pragma once
#include <memory>

namespace mlir {
class Pass;
class OpPassManager;

namespace db {
// DB Dialect Passes
std::unique_ptr<Pass> createEliminateNullsPass();
std::unique_ptr<Pass> createOptimizeRuntimeFunctionsPass();

// Pipeline functions
void createLowerDBPipeline(OpPassManager& pm);
void registerDBConversionPasses();

} // end namespace db
} // end namespace mlir
