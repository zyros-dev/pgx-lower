#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// RelAlg to DB Conversion Pass - Phase 4c-1 with Translator Pattern
//===----------------------------------------------------------------------===//

struct RelAlgToDBPass : public PassWrapper<RelAlgToDBPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDBPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::db::DBDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-db"; }
    StringRef getDescription() const final { return "Convert RelAlg dialect to DB dialect"; }
    
    // Check if an operation is a translation hook (top-level RelAlg operation)
    bool isTranslationHook(Operation* op) {
        return ::llvm::TypeSwitch<Operation*, bool>(op)
            .Case<::pgx::mlir::relalg::MaterializeOp>([&](Operation* op) {
                return true;  // MaterializeOp is a top-level translation hook
            })
            .Default([&](auto x) { return false; });
    }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass (Phase 4c-1 - Translator pattern)");
        
        auto funcOp = getOperation();
        ::pgx::mlir::relalg::TranslatorContext context;
        
        // Walk the function and translate RelAlg operations using Translator pattern
        funcOp.walk([&](Operation* op) {
            if (isTranslationHook(op)) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Found translation hook: " + op->getName().getStringRef().str());
                
                // Create translator for the operation
                auto translator = ::pgx::mlir::relalg::Translator::createTranslator(op);
                if (!translator) {
                    MLIR_PGX_WARNING("RelAlgToDB", "Failed to create translator for: " + 
                                     op->getName().getStringRef().str());
                    return;
                }
                
                // Set translator info (no consumer at top level, empty required attributes)
                translator->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
                
                // Set up builder at the operation location
                OpBuilder builder(op);
                
                // Execute the translation
                translator->produce(context, builder);
                translator->done();
                
                MLIR_PGX_DEBUG("RelAlgToDB", "Completed translation for: " + op->getName().getStringRef().str());
            }
        });
        
        MLIR_PGX_INFO("RelAlgToDB", "Completed RelAlg to DB conversion pass");
    }
};

} // namespace

namespace mlir {
namespace pgx_conversion {

std::unique_ptr<Pass> createRelAlgToDBPass() {
    return std::make_unique<RelAlgToDBPass>();
}

void registerRelAlgToDBConversionPasses() {
    PassRegistration<RelAlgToDBPass>();
}

} // namespace pgx_conversion
} // namespace mlir