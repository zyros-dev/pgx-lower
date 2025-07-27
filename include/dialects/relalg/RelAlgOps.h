#ifndef PGX_LOWER_RELALG_OPS_H
#define PGX_LOWER_RELALG_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Include the base dialect
#include "RelAlgDialect.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "dialects/subop/SubOpOps.h"

// Forward declarations
namespace pgx_lower::compiler::dialect::relalg {
class FunctionalDependencies;
class ColumnSet;
class ColumnFoldInfo;
} // namespace pgx_lower::compiler::dialect::relalg

// Define the Operator, PredicateOperator, etc. interfaces as empty for now
// These would normally come from RelAlgInterfaces.td
namespace pgx_lower::compiler::dialect::relalg {
namespace detail {
template<typename ConcreteOp>
class Operator {
public:
    // Empty interface for now
};

template<typename ConcreteOp>
class PredicateOperator {
public:
    // Empty interface for now
};

template<typename ConcreteOp>
class TupleLamdaOperator {
public:
    // Empty interface for now
};

template<typename ConcreteOp>
class UnaryOperator {
public:
    // Empty interface for now
};

template<typename ConcreteOp>
class BinaryOperator {
public:
    // Empty interface for now
};

template<typename ConcreteOp>
class ColumnFoldable {
public:
    // Empty interface for now
};
} // namespace detail
} // namespace pgx_lower::compiler::dialect::relalg

// Temporary trait definitions until we get the interfaces working
namespace mlir::OpTrait {
template<typename ConcreteOp>
using Operator = pgx_lower::compiler::dialect::relalg::detail::Operator<ConcreteOp>;

template<typename ConcreteOp>
using PredicateOperator = pgx_lower::compiler::dialect::relalg::detail::PredicateOperator<ConcreteOp>;

template<typename ConcreteOp>
using TupleLamdaOperator = pgx_lower::compiler::dialect::relalg::detail::TupleLamdaOperator<ConcreteOp>;

template<typename ConcreteOp>
using UnaryOperator = pgx_lower::compiler::dialect::relalg::detail::UnaryOperator<ConcreteOp>;

template<typename ConcreteOp>
using BinaryOperator = pgx_lower::compiler::dialect::relalg::detail::BinaryOperator<ConcreteOp>;
} // namespace mlir::OpTrait

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "RelAlgOps.h.inc"
#undef GET_OP_CLASSES

#endif // PGX_LOWER_RELALG_OPS_H