#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include <unordered_set>
#include <vector>

namespace pgx {
namespace mlir {
namespace relalg {

// ColumnSet represents a set of columns, typically used to track required/available columns
class ColumnSet {
private:
    std::unordered_set<const Column*> columns;
    
public:
    // Constructors
    ColumnSet() = default;
    ColumnSet(std::initializer_list<const Column*> cols) : columns(cols) {}
    
    // Insert a column
    void insert(const Column* col) {
        if (col) columns.insert(col);
    }
    
    // Remove a column
    void erase(const Column* col) {
        columns.erase(col);
    }
    
    // Check if a column is in the set
    bool contains(const Column* col) const {
        return columns.find(col) != columns.end();
    }
    
    // Check if empty
    bool empty() const {
        return columns.empty();
    }
    
    // Get size
    size_t size() const {
        return columns.size();
    }
    
    // Clear all columns
    void clear() {
        columns.clear();
    }
    
    // Set operations
    void insertAll(const ColumnSet& other) {
        for (const auto* col : other.columns) {
            columns.insert(col);
        }
    }
    
    // Iteration support
    auto begin() const { return columns.begin(); }
    auto end() const { return columns.end(); }
    
    // Convert to vector for ordered access
    std::vector<const Column*> toVector() const {
        return std::vector<const Column*>(columns.begin(), columns.end());
    }
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H