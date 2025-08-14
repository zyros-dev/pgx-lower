#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/RelAlgToDB/JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "execution/logging.h"
namespace {

class LowerToDBPass : public ::mlir::PassWrapper<LowerToDBPass, ::mlir::OperationPass<::mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-to-db"; }

   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect>();
      registry.insert<mlir::memref::MemRefDialect>();
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::db::DBDialect>();
      registry.insert<mlir::dsa::DSADialect>();
   }
   bool isTranslationHook(::mlir::Operation* op) {
      return ::llvm::TypeSwitch<::mlir::Operation*, bool>(op)
         .Case<mlir::relalg::MaterializeOp>([&](::mlir::Operation* op) {
            return true;
         })
         .Default([&](auto x) {
            return false;
         });
   }
   void runOnOperation() override {
      mlir::relalg::TranslatorContext loweringContext;
      PGX_DEBUG("RelAlg→DB Pass: Starting operation walk");
      
      // CRITICAL: Collect operations first to avoid iterator invalidation during erasure
      llvm::SmallVector<::mlir::Operation*> translationHooks;
      
      getOperation().walk([&](::mlir::Operation* op) {
         // Log all RelAlg operations found
         if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            PGX_DEBUG("RelAlg→DB Pass: Found RelAlg operation: " + op->getName().getStringRef().str());
         }
         
         if (isTranslationHook(op)) {
            translationHooks.push_back(op);
         }
      });
      
      PGX_INFO("RelAlg→DB Pass: Found " + std::to_string(translationHooks.size()) + " translation hooks to process");
      
      // Process collected operations safely
      for (auto* op : translationHooks) {
         PGX_INFO("RelAlg→DB Pass: Processing translation hook for: " + op->getName().getStringRef().str());
         
         // Add safety check for operation type
         if (!llvm::isa<mlir::relalg::MaterializeOp>(op)) {
            PGX_ERROR("RelAlg→DB Pass: Expected MaterializeOp but got: " + op->getName().getStringRef().str());
            continue;
         }
         
         PGX_DEBUG("RelAlg→DB Pass: Creating translator...");
         auto node = mlir::relalg::Translator::createTranslator(op);
         if (!node) {
            op->emitError("No translator found for operation: ") << op->getName();
            PGX_ERROR("RelAlg→DB Pass: No translator found for: " + op->getName().getStringRef().str());
            continue;
         }
         
         PGX_DEBUG("RelAlg→DB Pass: Setting translator info...");
         node->setInfo(nullptr, {});
         
         PGX_DEBUG("RelAlg→DB Pass: Creating OpBuilder...");
         ::mlir::OpBuilder builder(op);
         
         PGX_DEBUG("RelAlg→DB Pass: Calling produce on translator...");
         node->produce(loweringContext, builder);
         
         PGX_DEBUG("RelAlg→DB Pass: Calling done on translator...");
         node->done();
         
         PGX_INFO("RelAlg→DB Pass: Successfully translated: " + op->getName().getStringRef().str());
         
         // CRITICAL: Erase the RelAlg operation after successful translation
         // This prevents mixed dialect processing in subsequent passes
         PGX_DEBUG("RelAlg→DB Pass: Erasing translated operation: " + op->getName().getStringRef().str());
         op->erase();
      }
      
      PGX_DEBUG("RelAlg→DB Pass: Completed operation walk");
      
      // CRITICAL: Clean up any remaining RelAlg operations that were not translation hooks
      // These were processed indirectly through their parent MaterializeOp
      llvm::SmallVector<::mlir::Operation*> remainingRelAlgOps;
      getOperation().walk([&](::mlir::Operation* op) {
         if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            remainingRelAlgOps.push_back(op);
            PGX_INFO("RelAlg→DB Pass: Found remaining RelAlg operation to clean up: " + 
                     op->getName().getStringRef().str());
         }
      });
      
      // Erase remaining RelAlg operations
      for (auto* op : remainingRelAlgOps) {
         PGX_DEBUG("RelAlg→DB Pass: Erasing remaining RelAlg operation: " + 
                   op->getName().getStringRef().str());
         op->erase();
      }
      
      PGX_INFO("RelAlg→DB Pass: Cleaned up " + std::to_string(remainingRelAlgOps.size()) + 
               " remaining RelAlg operations");
      
      // Debug: Check what operations remain after translation
      PGX_DEBUG("RelAlg→DB Pass: Checking operations after translation");
      getOperation().walk([&](::mlir::Operation* op) {
         if (op->getDialect()) {
            std::string dialectName = op->getDialect()->getNamespace().str();
            if (dialectName == "relalg" || dialectName == "db" || dialectName == "dsa") {
               PGX_DEBUG("RelAlg→DB Pass: Found " + dialectName + " operation after translation: " + 
                         op->getName().getStringRef().str());
            }
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<mlir::Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }

void registerRelAlgConversionPasses(){
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createLowerToDBPass();
   });

   ::mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-relalg",
      "",
      createLowerRelAlgPipeline);
}
void createLowerRelAlgPipeline(mlir::OpPassManager& pm){
   pm.addNestedPass<::mlir::func::FuncOp>(mlir::relalg::createLowerToDBPass());
   pm.addPass(mlir::createCanonicalizerPass());
}

} // end namespace relalg
} // end namespace mlir

