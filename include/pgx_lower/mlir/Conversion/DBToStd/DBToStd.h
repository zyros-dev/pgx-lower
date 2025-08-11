#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace db {

// DBToStd conversion pass
std::unique_ptr<Pass> createLowerToStdPass();
void registerDBToStdConversion();

} // end namespace db
} // end namespace mlir
