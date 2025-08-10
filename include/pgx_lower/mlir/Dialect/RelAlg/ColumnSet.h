#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <unordered_set>
#include <vector>

namespace pgx {
namespace mlir {
namespace relalg {

// Forward declarations
class RelAlgDialect;

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
    
    // Insert another ColumnSet
    ColumnSet& insert(const ColumnSet& other) {
        for (const auto* col : other.columns) {
            columns.insert(col);
        }
        return *this;
    }
    
    // Remove a column
    void erase(const Column* col) {
        columns.erase(col);
    }
    
    // Remove columns from another set
    void remove(const ColumnSet& other) {
        for (const auto* col : other.columns) {
            columns.erase(col);
        }
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
    
    // Intersection
    ColumnSet intersect(const ColumnSet& other) const {
        ColumnSet result;
        for (const auto* col : columns) {
            if (other.contains(col)) {
                result.insert(col);
            }
        }
        return result;
    }
    
    // Check if sets have common elements
    bool intersects(const ColumnSet& other) const {
        for (const auto* col : columns) {
            if (other.contains(col)) {
                return true;
            }
        }
        return false;
    }
    
    // Check if this set is a subset of another
    bool isSubsetOf(const ColumnSet& other) const {
        for (const auto* col : columns) {
            if (!other.contains(col)) {
                return false;
            }
        }
        return true;
    }
    
    // Iteration support
    auto begin() const { return columns.begin(); }
    auto end() const { return columns.end(); }
    
    // Convert to vector for ordered access
    std::vector<const Column*> toVector() const {
        return std::vector<const Column*>(columns.begin(), columns.end());
    }
    
    // Convert to MLIR ArrayAttr
    ::mlir::ArrayAttr asRefArrayAttr(::mlir::MLIRContext* context);
    
    // Create from MLIR ArrayAttr
    static ColumnSet fromArrayAttr(::mlir::ArrayAttr arrayAttr);
    
    // Create from single column reference
    static ColumnSet from(ColumnRefAttr attrRef) {
        ColumnSet res;
        res.insert(&attrRef.getColumn());
        return res;
    }
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_COLUMNSET_H