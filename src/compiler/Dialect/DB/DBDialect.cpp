#include "compiler/Dialect/DB/DBDialect.h"

#include "compiler/Dialect/DB/DBOps.h"
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
// PostgreSQL: Simple attribute parsing implementation
mlir::Attribute pgx_lower::compiler::dialect::db::DBDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const {
   // TODO Phase 5: Implement proper attribute parsing for DB dialect
   // For now, defer to default parsing
   return nullptr;
}

// PostgreSQL: Simple attribute printing implementation  
void pgx_lower::compiler::dialect::db::DBDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const {
   // TODO Phase 5: Implement proper attribute printing for DB dialect
   // For now, use default printing
   printer << "db_attr";
}

#include "DBDialect.cpp.inc"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "DBTypes.cpp.inc"

// Operation definitions are in DBOps.cpp

// Register types
void pgx_lower::compiler::dialect::db::DBDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "DBTypes.cpp.inc"
    >();
}