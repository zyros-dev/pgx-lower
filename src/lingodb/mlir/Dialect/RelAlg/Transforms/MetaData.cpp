#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "runtime/Database.h"
namespace {
using mlir::relalg::Operator;
using mlir::relalg::BinaryOperator;
using mlir::relalg::UnaryOperator;
using mlir::relalg::TupleLamdaOperator;
using mlir::relalg::PredicateOperator;

class AttachMetaData : public ::mlir::PassWrapper<AttachMetaData, ::mlir::OperationPass<::mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   runtime::Database& db;
   public:
   AttachMetaData(runtime::Database& db):db(db){}
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         op.setMetaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),db.getTableMetaData(op.getTableIdentifier().str())));
      });
   }
};
class DetachMetaData : public ::mlir::PassWrapper<DetachMetaData, ::mlir::OperationPass<::mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-detach-meta-data"; }

   public:
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         getOperation().walk([&](mlir::relalg::BaseTableOp op) {
            op.setMetaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),std::make_shared<runtime::TableMetaData>()));
         });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<mlir::Pass> createAttachMetaDataPass(runtime::Database& db) { return std::make_unique<AttachMetaData>(db); }
std::unique_ptr<mlir::Pass> createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
} // end namespace relalg
} // end namespace mlir

