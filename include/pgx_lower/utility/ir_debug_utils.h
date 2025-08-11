//===- ir_debug_utils.h - MLIR IR debugging utilities -------------------===//
//
// Utility functions for debugging MLIR IR issues, especially circular
// references and infinite loops during IR traversal.
//
//===----------------------------------------------------------------------===//

#ifndef PGX_LOWER_UTILITY_IR_DEBUG_UTILS_H
#define PGX_LOWER_UTILITY_IR_DEBUG_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "execution/logging.h"
#include <set>
#include <string>

namespace utility {

/// Logs the module IR structure as plaintext for debugging.
/// 
/// @param module The MLIR module to print
/// @param label Descriptive label for the debug output
void logModuleIR(::mlir::ModuleOp module, const std::string& label);

/// Detects circular IR structures that cause infinite walk() loops.
/// Uses manual traversal with proper path tracking instead of walk().
/// 
/// @param module The MLIR module to check for circular references
/// @return true if circular IR is detected, false if IR is well-formed
bool hasCircularIR(::mlir::ModuleOp module);

} // namespace utility

#endif // PGX_LOWER_UTILITY_IR_DEBUG_UTILS_H