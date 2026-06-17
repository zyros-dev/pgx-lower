#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsAttributes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;
using namespace ::mlir::db;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
#define GET_ATTRDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsAttributes.cpp.inc"
void DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/mlir/Dialect/DB/IR/DBOps.cpp.inc"
      >();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/mlir/Dialect/DB/IR/DBOpsAttributes.cpp.inc"
       >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   runtimeFunctionRegistry = mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext());
}
#include "lingodb/mlir/Dialect/DB/IR/DBOpsDialect.cpp.inc"
