#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/db/DBDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::relalg;

void RelAlgDialect::initialize() {
    // Add types from TableGen
    addTypes<
#define GET_TYPEDEF_LIST
#include "RelAlgTypes.h.inc"
    >();
    
    // Add operations from TableGen
    addOperations<
#define GET_OP_LIST
#include "RelAlgOps.cpp.inc"
    >();
    
    // Load dependent dialects
    getContext()->loadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
    getContext()->loadDialect<mlir::arith::ArithDialect>();
    getContext()->loadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
}

#include "RelAlgDialect.cpp.inc"