#include "dialects/db/DBDialect.h"

#include "dialects/db/DBOps.h"
// #include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h" // TODO: Port if needed
// #include "lingodb/compiler/mlir-support/tostring.h" // TODO: Port if needed

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
using namespace mlir;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void mlir::db::DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "DBOps.cpp.inc"

      >();
   addInterfaces<DBInlinerInterface>();
   // registerTypes(); // TODO: Implement if needed
   // runtimeFunctionRegistry = db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext()); // TODO: Port if needed
}

// TODO: Implement materializeConstant when needed
// ::mlir::Operation* mlir::db::DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
//    // TODO: Port decimal/date string conversion functions if needed
//    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) {
//       return builder.create<mlir::db::ConstantOp>(loc, type, value);
//    }
//    return nullptr;
// }
#include "DBDialect.cpp.inc"
