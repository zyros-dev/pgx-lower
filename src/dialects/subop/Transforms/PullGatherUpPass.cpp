#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/Transforms/ColumnUsageAnalysis.h"
#include "dialects/subop/Transforms/Passes.h"
#include "dialects/subop/Transforms/SubOpDependencyAnalysis.h"
#include "dialects/tuplestream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"

namespace {
using namespace pgx_lower::compiler::dialect;

class PullGatherUpPass : public mlir::PassWrapper<PullGatherUpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PullGatherUpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-pull-gather-up"; }

   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();

      std::vector<subop::GatherOp> gatherOps;
      getOperation()->walk([&](subop::GatherOp gatherOp) {
         gatherOps.push_back(gatherOp);
      });
      for (auto gatherOp : gatherOps) {
         std::vector<mlir::NamedAttribute> remaining = gatherOp.getMapping().getValue();
         gatherOp.getRes().replaceAllUsesWith(gatherOp.getStream());
         auto* currentChild = gatherOp.getStream().getDefiningOp();
         mlir::Block* safeBlock = gatherOp->getBlock();
         while (currentChild) {
            if (auto refProducer = mlir::dyn_cast_or_null<subop::ReferenceProducer>(currentChild)) {
               // TODO Phase 4: Fix Column comparison - getColumn() returns reference, not pointer
               auto& refCol = refProducer.getProducedReference().getColumn();
               auto& gathCol = gatherOp.getRef().getColumn();
               if (&refCol == &gathCol) {
                  mlir::Block* minimalSafeBlock = nullptr;
                  for (auto operand : currentChild->getOperands()) {
                     if (!mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                        mlir::Block* localSafeBlock;
                        if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                           localSafeBlock = blockArgument.getOwner()->getParentOp()->getBlock();
                        } else if (auto* definingOp = operand.getDefiningOp()) {
                           localSafeBlock = definingOp->getBlock();
                        }
                        if (minimalSafeBlock == nullptr || minimalSafeBlock->getParentOp()->isAncestor(localSafeBlock->getParentOp())) {
                           minimalSafeBlock = localSafeBlock;
                        }
                     }
                  }
                  if (minimalSafeBlock != nullptr) {
                     safeBlock = minimalSafeBlock;
                  }
                  break;
               }
            }
            currentChild = currentChild->getNumOperands() == 1 ? currentChild->getOperand(0).getDefiningOp() : nullptr;
         }
         mlir::Value currStream = gatherOp.getStream();
         gatherOp->setOperand(0, gatherOp.getResult());
         mlir::Operation* currentParent;
         mlir::Value lastStream;
         while (currStream) {
            auto users = currStream.getUsers();
            if (users.begin() == users.end()) break;
            auto second = users.begin();
            second++;
            if (second != users.end()) break;
            currentParent = *users.begin();
            bool otherStreams = false;
            for (auto v : currentParent->getOperands()) {
               otherStreams |= v != currStream && mlir::isa<tuples::TupleStreamType>(v.getType());
            }
            if (otherStreams) break;
            auto usedColumns = columnUsageAnalysis.getUsedColumns(currentParent);
            std::vector<mlir::NamedAttribute> usedByCurrent;
            std::vector<mlir::NamedAttribute> notUsedByCurrent;
            for (auto x : remaining) {
               // TODO Phase 4: Fix column comparison - getColumn() returns reference, not pointer
               auto& colRef = mlir::cast<tuples::ColumnDefAttr>(x.getValue()).getColumn();
               auto* colPtr = &colRef;
               if (colPtr && usedColumns.find(const_cast<tuples::Column*>(colPtr)) != usedColumns.end()) {
                  usedByCurrent.push_back(x);
               } else {
                  notUsedByCurrent.push_back(x);
               }
            }
            if (!usedByCurrent.empty()) {
               mlir::OpBuilder builder(currentParent);
               auto newGatherOp = builder.create<subop::GatherOp>(gatherOp->getLoc(), currStream, gatherOp.getRef(), builder.getDictionaryAttr(usedByCurrent));
               currStream.replaceAllUsesWith(newGatherOp.getResult());
               newGatherOp->setOperand(0, currStream);
               lastStream = newGatherOp.getResult();
               //newGatherOp->dump();
            } else {
               lastStream = currStream;
            }
            remaining = std::move(notUsedByCurrent);

            currStream = currentParent->getNumResults() == 1 ? currentParent->getResult(0) : mlir::Value();
         }
         if (!currStream && currentParent) {
            if (mlir::isa<tuples::ReturnOp>(currentParent)) {
               if (auto nestedMapOp = mlir::dyn_cast_or_null<subop::NestedMapOp>(currentParent->getParentOp())) {
                  if (safeBlock->getParentOp()->isProperAncestor(currentParent->getParentOp())) {
                     currStream = nestedMapOp.getResult();
                  } else {
                     currStream = lastStream;
                  }
               } else {
                  currStream = lastStream;
               }
            }
         }
         if (!remaining.empty() && currStream) {
            mlir::OpBuilder builder(currStream.getContext());
            builder.setInsertionPointAfter(currStream.getDefiningOp());
            if (!currStream.getUsers().empty()) {
               auto newGatherOp = builder.create<subop::GatherOp>(gatherOp->getLoc(), currStream, gatherOp.getRef(), gatherOp.getMapping());
               currStream.replaceAllUsesWith(newGatherOp.getResult());
               newGatherOp->setOperand(0, currStream);
            }
         }
         gatherOp->dropAllReferences();
         gatherOp->erase();
      }

      //getOperation()->dump();
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createPullGatherUpPass() { return std::make_unique<PullGatherUpPass>(); }