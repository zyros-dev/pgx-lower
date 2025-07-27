#include "dialects/pg/PgDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pg;

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

void PgDialect::initialize() {
    // TableGen generated operation and type registration
    addOperations<
#define GET_OP_LIST
#include "PgDataAccess.cpp.inc"
        >();
    
    addOperations<
#define GET_OP_LIST
#include "PgPolymorphic.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "PgTypes.cpp.inc"
        >();
}

PgDialect::PgDialect(MLIRContext *context)
: Dialect(getDialectNamespace(), context, TypeID::get<PgDialect>()) {
    initialize();
}

// Include TableGen generated definitions
#define GET_OP_CLASSES
#include "PgDataAccess.cpp.inc"

#define GET_OP_CLASSES
#include "PgPolymorphic.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PgTypes.cpp.inc"