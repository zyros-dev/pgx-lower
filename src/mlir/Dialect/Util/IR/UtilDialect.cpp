#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/FunctionHelper.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>

struct UtilInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
};
void pgx::mlir::util::UtilDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Util/IR/UtilOps.cpp.inc"
      >();
   addInterfaces<UtilInlinerInterface>();
   registerTypes();
   functionHelper = std::make_shared<::pgx::mlir::util::FunctionHelper>();
}
#include "mlir/Dialect/Util/IR/UtilOpsDialect.cpp.inc"
