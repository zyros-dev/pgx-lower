#pragma once

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;

namespace db {

std::unique_ptr<Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

}
}
