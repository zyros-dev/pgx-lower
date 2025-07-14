//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Base Definition
//===----------------------------------------------------------------------===//

#ifndef PG_DIALECT
#define PG_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Definition
//===----------------------------------------------------------------------===//

def Pg_Dialect : Dialect {
    let name = "pg";
    let summary = "PostgreSQL query compilation dialect";

    let cppNamespace = "::mlir::pg";
    let useDefaultTypePrinterParser = 1;
}

#endif // PG_DIALECT