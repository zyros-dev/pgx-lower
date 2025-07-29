#include "dialects/db/DBDialect.h"

#include "dialects/db/DBOps.h"
// #include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h" // TODO Phase 5: Port if needed
// #include "lingodb/compiler/mlir-support/tostring.h" // TODO Phase 5: Port if needed

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace pgx_lower::compiler::dialect;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void pgx_lower::compiler::dialect::db::DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "DBOps.cpp.inc"

      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   // runtimeFunctionRegistry = db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext()); // TODO Phase 5: Port if needed
}

// TODO Phase 5: Implement materializeConstant when needed
// ::mlir::Operation* mlir::db::DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
//    // TODO Phase 9: Port decimal/date string conversion functions if needed
//    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) {
//       return builder.create<mlir::db::ConstantOp>(loc, type, value);
//    }
//    return nullptr;
// }
#include "DBDialect.cpp.inc"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "DBTypes.cpp.inc"

// Operation definitions
#define GET_OP_CLASSES
#include "DBOps.cpp.inc"

// Register types
void pgx_lower::compiler::dialect::db::DBDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "DBTypes.cpp.inc"
    >();
}