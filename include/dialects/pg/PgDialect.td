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
    let description = [{
        The PostgreSQL dialect represents high-level SQL operations that can be
        progressively lowered to standard MLIR dialects for efficient compilation.
        
        This dialect provides operations for:
        - Table scanning and tuple iteration
        - Field access with PostgreSQL type safety
        - Predicates and filtering
        - Aggregation operations
        - Null handling
    }];
    
    let cppNamespace = "::mlir::pg";
    let useDefaultTypePrinterParser = 1;
}

#endif // PG_DIALECT