#include "dialects/subop/SubOpDialect.h"

#include "dialects/db/DBDialect.h"
#include "dialects/subop/SubOpOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

struct SubOperatorInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned, mlir::IRMapping& valueMapping) const override {
      return true;
   }
};
struct SubOpFoldInterface : public mlir::DialectFoldInterface {
   using DialectFoldInterface::DialectFoldInterface;

   bool shouldMaterializeInto(mlir::Region* region) const final {
      return true;
   }
};
void subop::SubOpDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "SubOpOps.cpp.inc"

      >();
   registerTypes();
   registerAttrs();
   addInterfaces<SubOperatorInlinerInterface>();
   addInterfaces<SubOpFoldInterface>();
   getContext()->loadDialect<db::DBDialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   getContext()->loadDialect<mlir::index::IndexDialect>();
   getContext()->loadDialect<tuples::TupleStreamDialect>();
}

#include "SubOpDialect.cpp.inc"
