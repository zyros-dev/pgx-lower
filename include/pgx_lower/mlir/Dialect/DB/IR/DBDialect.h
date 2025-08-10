#ifndef PGX_DIALECT_DB_IR_DBDIALECT_H
#define PGX_DIALECT_DB_IR_DBDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

// Forward declare the implementation classes to resolve namespace issues
namespace mlir {
class Type;
class DialectAsmParser;
class DialectAsmPrinter;
}

#include "mlir/Dialect/DB/IR/DBOpsDialect.h.inc"

#endif // PGX_DIALECT_DB_IR_DBDIALECT_H