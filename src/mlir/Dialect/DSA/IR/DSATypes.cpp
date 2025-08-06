#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"