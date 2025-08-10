#ifndef PGX_DIALECT_UTIL_IR_UTILOPS_H
#define PGX_DIALECT_UTIL_IR_UTILOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include <optional>

#include "mlir/Dialect/Util/IR/UtilDialect.h"

// Add proper namespace imports for Properties system
namespace pgx {
namespace mlir {
using ::mlir::Attribute;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::PatternRewriter;
} // namespace mlir
} // namespace pgx

#include "mlir/Dialect/Util/IR/UtilTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Util/IR/UtilOps.h.inc"

#endif // PGX_DIALECT_UTIL_IR_UTILOPS_H