#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::tuples;

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
   
   // Add attributes from TableGen
   addAttributes<
#define GET_ATTRDEF_LIST
#include "TupleStreamAttrs.h.inc"
   >();
}

#include "TupleStreamDialect.cpp.inc"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "TupleStreamTypes.cpp.inc"

// Attribute definitions
#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.cpp.inc"