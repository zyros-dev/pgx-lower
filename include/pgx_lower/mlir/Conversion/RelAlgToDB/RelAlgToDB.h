#ifndef MLIR_CONVERSION_RELALGTODB_RELALGTODB_H
#define MLIR_CONVERSION_RELALGTODB_RELALGTODB_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class Pass;

namespace pgx_conversion {

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

/// Create the RelAlg to DB conversion pass
/// Phase 4c-0: Cleaned up for Translator pattern implementation
std::unique_ptr<Pass> createRelAlgToDBPass();

/// Register RelAlg to DB conversion passes
void registerRelAlgToDBConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODB_H