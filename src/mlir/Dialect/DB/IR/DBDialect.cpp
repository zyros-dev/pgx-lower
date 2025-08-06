#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::db;

struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};

void DBDialect::initialize() {
    PGX_DEBUG("Initializing DB dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
    >();
    
    addInterfaces<DBInlinerInterface>();
    
    PGX_DEBUG("DB dialect initialization complete");
}

#include "mlir/Dialect/DB/IR/DBOpsDialect.cpp.inc"