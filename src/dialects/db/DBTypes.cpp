#include "dialects/db/DBTypes.h"
#include "dialects/db/DBDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "DBTypes.cpp.inc"

namespace pgx_lower::compiler::dialect::db {

void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "DBTypes.cpp.inc"
      >();
}

} // namespace pgx_lower::compiler::dialect::db