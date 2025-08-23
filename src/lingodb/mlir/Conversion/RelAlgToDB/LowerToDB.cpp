#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/mlir/Conversion/RelAlgToDB/JoinTranslator.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "pgx-lower/execution/logging.h"
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
      
      getOperation().walk([&](::mlir::Operation* op) {
         if (isTranslationHook(op)) {
            PGX_INFO("LowerToDB: Found translation hook for " + op->getName().getStringRef().str());
            auto node = mlir::relalg::Translator::createTranslator(op);
            if (!node) {
               op->emitError("No translator found for operation: ") << op->getName();
               return;
            }
            
            PGX_INFO("LowerToDB: Calling setInfo and produce on translator");
            node->setInfo(nullptr, {});
            ::mlir::OpBuilder builder(op);
            node->produce(loweringContext, builder);
            node->done();
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

