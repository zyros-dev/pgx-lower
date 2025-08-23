#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir {



namespace db {
} // end namespace db
} // end namespace mlir
#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace mlir::db {
void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::db
