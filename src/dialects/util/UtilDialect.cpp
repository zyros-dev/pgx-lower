#include "dialects/util/UtilDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::util;

struct UtilInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
};

void UtilDialect::initialize() {
   // Add operations if we define any
   // addOperations<
   // #define GET_OP_LIST
   // #include "UtilOps.cpp.inc"
   //    >();
   
   addInterfaces<UtilInlinerInterface>();
   registerTypes();
}

void UtilDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "UtilTypes.cpp.inc"
    >();
}

// Type method implementations
mlir::Type mlir::util::BufferType::getElementType() {
    return getT();
}

#include "UtilDialect.cpp.inc"
