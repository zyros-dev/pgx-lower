#include "compiler/Dialect/util/UtilDialect.h"
#include "compiler/Dialect/util/UtilTypes.h"
#include "compiler/Dialect/util/UtilOps.h"
// Include FunctionHelper before generated code
#include "compiler/Dialect/util/FunctionHelper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

// Include interface definitions
#include "UtilOpsTypeInterfaces.cpp.inc"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::util;

struct UtilInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
};

void UtilDialect::initialize() {
   // Add types
   addTypes<
#define GET_TYPEDEF_LIST
#include "UtilTypes.cpp.inc"
   >();
   
   // Add operations
   addOperations<
   #define GET_OP_LIST
   #include "UtilOps.cpp.inc"
      >();
   
   addInterfaces<UtilInlinerInterface>();
   // Initialize FunctionHelper after TableGen includes are processed
}

void UtilDialect::registerTypes() {
    // Handled in initialize()
}

// Include TableGen generated definitions
#include "UtilDialect.cpp.inc"

// getFunctionHelper implementation
namespace pgx_lower::compiler::dialect::util {
FunctionHelper& UtilDialect::getFunctionHelper() {
    if (!functionHelper) {
        functionHelper = std::make_shared<FunctionHelper>();
    }
    return *functionHelper;
}
}

#define GET_TYPEDEF_CLASSES
#include "UtilTypes.cpp.inc"

// Type method implementations
mlir::Type pgx_lower::compiler::dialect::util::BufferType::getElementType() {
    return getT();
}
