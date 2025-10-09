#ifndef MLIR_CONVERSION_RELALGTODB_RELALGTODB_H
#define MLIR_CONVERSION_RELALGTODB_RELALGTODB_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class Pass;

namespace pgx_conversion {
std::unique_ptr<Pass> createRelAlgToDBPass();
void registerRelAlgToDBConversionPasses();
} // namespace pgx_conversion
} // namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODB_H