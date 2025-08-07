#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

struct DSAInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};

// Type definitions are included in DSATypes.cpp, not here

void DSADialect::initialize() {
    PGX_DEBUG("Initializing DSA dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
    >();
    
    registerTypes();
    
    addInterfaces<DSAInlinerInterface>();
    
    PGX_DEBUG("DSA dialect initialization complete");
}

#include "mlir/Dialect/DSA/IR/DSAOpsDialect.cpp.inc"