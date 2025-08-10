#ifndef MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
#define MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H

#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include <vector>

namespace pgx::mlir::relalg {

/// Represents functional dependencies between columns
class FunctionalDependencies {
public:
  FunctionalDependencies() = default;
  
  /// Check if this FD set is empty
  bool empty() const { return fds.empty(); }
  
  /// Clear all functional dependencies
  void clear() { fds.clear(); }
  
  /// Insert all functional dependencies from another set
  void insert(const FunctionalDependencies& other) {
    fds.insert(fds.end(), other.fds.begin(), other.fds.end());
  }
  
  /// Insert a functional dependency
  void insert(const ColumnSet& left, const ColumnSet& right) {
    fds.push_back({left, right});
  }
  
  /// Reduce keys based on functional dependencies
  ColumnSet reduce(const ColumnSet& keys) {
    ColumnSet res = keys;
    ColumnSet remove;
    for (auto fd : fds) {
      if (fd.first.isSubsetOf(keys)) {
        remove.insert(fd.second);
      }
    }
    res.remove(remove);
    return res;
  }
  
private:
  std::vector<std::pair<ColumnSet, ColumnSet>> fds;
};

} // namespace pgx::mlir::relalg

#endif // MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H