//===- DSAToStd.h - DSA to Standard dialects conversion pass -----*- C++ -*-===//

#ifndef MLIR_CONVERSION_DSATOSTD_DSATOSTD_H
#define MLIR_CONVERSION_DSATOSTD_DSATOSTD_H

#include <memory>

namespace mlir {
class Pass;

/// Create a pass to convert DSA dialect operations to Standard MLIR dialects
/// with runtime function calls (Phase 4e-2).
/// 
/// Converts DSA operations for data structure management:
/// - dsa.create_ds → memref allocation + runtime call to create table builder
/// - dsa.ds_append → nullable extraction + runtime append calls
/// - dsa.next_row → runtime call to finalize current row
std::unique_ptr<Pass> createDSAToStdPass();

} // namespace mlir

#endif // MLIR_CONVERSION_DSATOSTD_DSATOSTD_H