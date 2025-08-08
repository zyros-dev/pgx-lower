#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "execution/logging.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::util;

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Util/IR/UtilOpsTypes.cpp.inc"

namespace pgx {
namespace mlir {
namespace util {
void UtilDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Util/IR/UtilOpsTypes.cpp.inc"
    >();
}
} // namespace util
} // namespace mlir
} // namespace pgx