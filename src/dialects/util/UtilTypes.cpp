#include "llvm/ADT/TypeSwitch.h"

#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "dialects/util/UtilTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include <unordered_set>

using namespace pgx_lower::compiler::dialect;

// Custom method implementations (if needed)
mlir::Type util::BufferType::getElementType() {
   return util::RefType::get(getContext(), getT());
}

#include "UtilOpsTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "UtilTypes.cpp.inc"

namespace pgx_lower::compiler::dialect::util {
void UtilDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "UtilTypes.cpp.inc"

      >();
}

} // namespace pgx_lower::compiler::dialect::util
