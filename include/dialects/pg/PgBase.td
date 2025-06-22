//===-- PgBase.td - PostgreSQL dialect base definitions ---*- tablegen -*-===//
//
// PostgreSQL dialect base TableGen definitions
//
//===----------------------------------------------------------------------===//

#ifndef PG_BASE
#define PG_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "dialects/pg/PgDialect.td"

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Forward Declaration
//===----------------------------------------------------------------------===//

// Dialect definition is in PgDialect.td to avoid duplication

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class Pg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pg_Dialect, mnemonic, traits>;

class Pg_Type<string name, string typeMnemonic> : TypeDef<Pg_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif // PG_BASE