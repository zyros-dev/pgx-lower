#ifndef PGX_LOWER_RELALG_INTERFACES_H
#define PGX_LOWER_RELALG_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "dialects/tuplestream/Column.h"
#include <set>
#include <functional>

// Forward declarations and definitions
namespace pgx_lower::compiler::dialect::relalg {

// ColumnSet represents a set of columns used in relational operations
class ColumnSet {
private:
    std::set<std::shared_ptr<tuples::Column>> columns_;
    
public:
    ColumnSet() = default;
    
    void insert(std::shared_ptr<tuples::Column> col) {
        columns_.insert(col);
    }
    
    bool contains(std::shared_ptr<tuples::Column> col) const {
        return columns_.find(col) != columns_.end();
    }
    
    size_t size() const { return columns_.size(); }
    bool empty() const { return columns_.empty(); }
};

// FunctionalDependencies tracks dependencies between columns
class FunctionalDependencies {
public:
    FunctionalDependencies() = default;
    // Minimal implementation for now
};

// ColumnFoldInfo for column folding optimization
class ColumnFoldInfo {
public:
    ColumnFoldInfo() = default;
    // Minimal implementation for now
};
namespace detail {
// Helper functions that will be defined in the implementation
ColumnSet getUsedColumns(mlir::Operation* op);
ColumnSet getAvailableColumns(mlir::Operation* op);
ColumnSet getCreatedColumns(mlir::Operation* op);
bool canColumnReach(mlir::Operation* op, mlir::Operation* source, mlir::Operation* target, const void* col);
ColumnSet getSetOpCreatedColumns(mlir::Operation* op);
ColumnSet getSetOpUsedColumns(mlir::Operation* op);
FunctionalDependencies getFDs(mlir::Operation* op);
void moveSubTreeBefore(mlir::Operation* op, mlir::Operation* before);
ColumnSet getFreeColumns(mlir::Operation* op);
std::pair<mlir::Type, mlir::Type> getBinaryOperatorType(mlir::Operation* op);
mlir::Type getUnaryOperatorType(mlir::Operation* op);

// Global sets for operator properties
extern std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> assoc;
extern std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> lAsscom;
extern std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> rAsscom;
extern std::set<std::pair<mlir::Type, mlir::Type>> reorderable;
extern std::set<std::pair<mlir::Type, mlir::Type>> lPushable;
extern std::set<std::pair<mlir::Type, mlir::Type>> rPushable;

// Predicate helpers
void addPredicate(mlir::Operation* op, mlir::Value pred);
void addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> producer);
void initPredicate(mlir::Operation* op);
} // namespace detail
} // namespace pgx_lower::compiler::dialect::relalg

// Include generated interface declarations
#include "RelAlgInterfaces.h.inc"

#endif // PGX_LOWER_RELALG_INTERFACES_H