#ifndef PGX_DIALECT_DB_IR_DBTYPES_H
#define PGX_DIALECT_DB_IR_DBTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Include the dialect first to forward-declare namespace
namespace pgx { namespace db { class DBDialect; } }

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif // PGX_DIALECT_DB_IR_DBTYPES_H