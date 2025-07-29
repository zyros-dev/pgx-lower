#ifndef PGX_LOWER_RELALG_INTERFACES_H
#define PGX_LOWER_RELALG_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include <set>
#include <functional>

// Forward declarations and definitions
#include "ColumnSet.h"
namespace pgx_lower::compiler::dialect::relalg {

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

// Operation inlining helper
void inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Block* newBlock, mlir::IRMapping& mapping, mlir::Operation* first = nullptr);
} // namespace detail
} // namespace pgx_lower::compiler::dialect::relalg

// Include generated interface declarations
#include "RelAlgInterfaces.h.inc"

#endif // PGX_LOWER_RELALG_INTERFACES_H