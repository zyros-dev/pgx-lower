#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

// All DSA operations use auto-generated parser/printer from assemblyFormat in TableGen

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"