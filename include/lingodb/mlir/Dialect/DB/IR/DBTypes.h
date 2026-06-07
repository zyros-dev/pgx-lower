#ifndef MLIR_DIALECT_DB_IR_DBTYPES_H
#define MLIR_DIALECT_DB_IR_DBTYPES_H

#include "postgres_ext.h"
#include "catalog/pg_type_d.h"

#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace mlir::db {
using PgOid = Oid;

bool isPgValueType(mlir::Type type);
PgOid getPgTypeOid(mlir::Type type);
int32_t getPgTypmod(mlir::Type type);
PgOid getPgCollation(mlir::Type type);
PgNullability getPgNullability(mlir::Type type);
mlir::Type withPgNullability(mlir::Type type, PgNullability nullability);
mlir::Type getPgPhysicalCarrierType(mlir::Type type);
} // namespace mlir::db

#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif // MLIR_DIALECT_DB_IR_DBTYPES_H
