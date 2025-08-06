#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "execution/logging.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace pgx::db;

//===----------------------------------------------------------------------===//
// DB Dialect Operations
//===----------------------------------------------------------------------===//

// Custom assembly format implementations would go here
// For now, we'll use the default format

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"