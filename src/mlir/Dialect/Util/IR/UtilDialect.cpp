#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "execution/logging.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace pgx::mlir::util;

//===----------------------------------------------------------------------===//
// Util dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Util/IR/UtilOpsDialect.cpp.inc"

void UtilDialect::initialize() {
    MLIR_PGX_DEBUG("Util", "Initializing Util dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Util/IR/UtilOps.cpp.inc"
    >();
    
    registerTypes();
}