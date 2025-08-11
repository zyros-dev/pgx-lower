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

/// Create the RelAlg to mixed DB+DSA conversion pass
/// Phase 4d: PostgreSQL SPI integration with streaming producer-consumer pattern
/// Generates mixed DB operations (PostgreSQL table access) and DSA operations (result materialization)
std::unique_ptr<Pass> createRelAlgToDBPass();

/// Register RelAlg to mixed DB+DSA conversion passes (PostgreSQL SPI architecture)
void registerRelAlgToDBConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODB_H