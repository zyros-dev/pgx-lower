#pragma once

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;

namespace db {

// DBToStd conversion pass
std::unique_ptr<Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

// Minimal test pass for debugging
std::unique_ptr<Pass> createMinimalDBToStdPass();

} // end namespace db
} // end namespace mlir
