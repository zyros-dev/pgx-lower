#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir {
namespace relalg {
} // end namespace relalg
} // end namespace mlir

#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"

namespace mlir::relalg {

void RelAlgDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::relalg