#ifndef MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
#define MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::relalg {

// Forward declarations
class ColumnSet;

/// Represents functional dependencies between columns
/// This is a minimal stub implementation for LLVM 20 compatibility
class FunctionalDependencies {
public:
  FunctionalDependencies() = default;
  
  /// Check if this FD set is empty
  bool empty() const { return dependencies.empty(); }
  
  /// Clear all functional dependencies
  void clear() { dependencies.clear(); }
  
private:
  // Simple representation: map from determinant columns to dependent columns
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> dependencies;
};

} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H