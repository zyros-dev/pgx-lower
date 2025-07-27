//===----------------------------------------------------------------------===//
// RelAlg Dialect Implementation
//===----------------------------------------------------------------------===//

#include "dialects/relalg/RelAlgDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::relalg;

//===----------------------------------------------------------------------===//
// RelAlg dialect.
//===----------------------------------------------------------------------===//

void RelAlgDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "RelAlgOps.cpp.inc"
    >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "RelAlgTypes.cpp.inc"
    >();
}

#include "RelAlgDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RelAlg type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "RelAlgTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// RelAlg operation definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RelAlgOps.cpp.inc"