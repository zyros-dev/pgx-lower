#ifndef PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H
#define PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace pgx_conversion {

/// Create a pass that lowers RelAlg dialect operations to DSA dialect operations.
/// This pass converts:
/// - relalg.basetable -> dsa.scan_source  
/// - relalg.materialize -> DSA result builder pattern (create_ds, ds_append, next_row, finalize)
/// - relalg.return -> proper terminator handling
std::unique_ptr<Pass> createRelAlgToDSAPass();

/// Register RelAlg to DSA conversion passes
void registerRelAlgToDSAConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H