#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/db/DBDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::relalg;

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
    getContext()->loadDialect<mlir::db::DBDialect>();
    getContext()->loadDialect<mlir::arith::ArithDialect>();
    getContext()->loadDialect<mlir::tuples::TupleStreamDialect>();
}

#include "RelAlgDialect.cpp.inc"