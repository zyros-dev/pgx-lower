#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"