#ifndef MLIR_DIALECT_DB_IR_DBTYPES_H
#define MLIR_DIALECT_DB_IR_DBTYPES_H

#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace mlir::db {
using PgOid = std::uint32_t;

inline constexpr PgOid kPgInvalidOid = 0;
inline constexpr PgOid kPgBoolOid = 16;
inline constexpr PgOid kPgInt8Oid = 20;
inline constexpr PgOid kPgInt2Oid = 21;
inline constexpr PgOid kPgInt4Oid = 23;
inline constexpr PgOid kPgTextOid = 25;
inline constexpr PgOid kPgFloat4Oid = 700;
inline constexpr PgOid kPgFloat8Oid = 701;
inline constexpr PgOid kPgBpcharOid = 1042;
inline constexpr PgOid kPgVarcharOid = 1043;
inline constexpr PgOid kPgDateOid = 1082;
inline constexpr PgOid kPgTimestampOid = 1114;
inline constexpr PgOid kPgIntervalOid = 1186;
inline constexpr PgOid kPgNumericOid = 1700;

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
