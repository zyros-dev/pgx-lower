#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

// BaseTableOp uses auto-generated parser/printer from assemblyFormat in TableGen




// MaterializeOp uses auto-generated parser/printer from assemblyFormat in TableGen

// GetColumnOp uses auto-generated parser/printer from assemblyFormat in TableGen

// Properties system disabled - using traditional attributes

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"