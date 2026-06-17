#ifndef MLIR_DIALECT_DB_IR_DBTYPES_H
#define MLIR_DIALECT_DB_IR_DBTYPES_H

#include "lingodb/mlir/Dialect/DB/IR/DBOpsAttributes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "lingodb/mlir/Dialect/DB/IR/DBPgTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace mlir::db {

enum class PgRowFieldLookupStatus { NotFound, Found, Ambiguous };

class PgRowFieldLookupResult {
  public:
    static PgRowFieldLookupResult notFound();
    static PgRowFieldLookupResult found(PgRowFieldAttr field);
    static PgRowFieldLookupResult ambiguous();

    bool isFound() const;
    bool isAmbiguous() const;
    uint32_t getFieldIndex() const;
    PgRowFieldAttr getField() const;

  private:
    PgRowFieldLookupResult(PgRowFieldLookupStatus status, PgRowFieldAttr field);

    PgRowFieldLookupStatus status;
    PgRowFieldAttr field;
};

bool isPgValueType(mlir::Type type);
PgOid getPgTypeOid(mlir::Type type);
int32_t getPgTypmod(mlir::Type type);
PgOid getPgCollation(mlir::Type type);
PgNullability getPgNullability(mlir::Type type);
mlir::Type withPgNullability(mlir::Type type, PgNullability nullability);
mlir::Type getPgPhysicalCarrierType(mlir::Type type);

bool isPgRowType(mlir::Type type);
bool isPgRowStreamType(mlir::Type type);
PgRowSchemaAttr getPgRowSchema(mlir::Type type);
uint32_t getPgRowFieldCount(mlir::Type type);
PgRowFieldAttr getPgRowFieldByIndex(mlir::Type type, uint32_t index);
PgRowFieldLookupResult lookupPgRowFieldBySource(mlir::Type type, uint32_t varno, int16_t attno);
mlir::Type getPgRowFieldType(PgRowFieldAttr field);
PgOid getPgRowFieldOid(PgRowFieldAttr field);
int32_t getPgRowFieldTypmod(PgRowFieldAttr field);
PgOid getPgRowFieldCollation(PgRowFieldAttr field);
PgNullability getPgRowFieldNullability(PgRowFieldAttr field);
} // namespace mlir::db

#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif // MLIR_DIALECT_DB_IR_DBTYPES_H
