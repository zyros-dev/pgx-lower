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

} // end namespace db
} // end namespace mlir
