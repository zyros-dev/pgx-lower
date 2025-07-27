#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tuples;

void TupleStreamDialect::initialize() {
   // addOperations<
   //    // Add operations when needed
   // >();
   
   // TODO: Load ColumnManager if needed
   // Initialize types from TableGen
   addTypes<
#define GET_TYPEDEF_LIST
#include "TupleStreamTypes.h.inc"
   >();
   
   // TODO: Add attributes when needed
}

#include "TupleStreamDialect.cpp.inc"