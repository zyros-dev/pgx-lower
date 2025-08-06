#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "execution/logging.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::db;

//===----------------------------------------------------------------------===//
// DB Dialect Types
//===----------------------------------------------------------------------===//

// Generate the type storage classes and type implementations
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"