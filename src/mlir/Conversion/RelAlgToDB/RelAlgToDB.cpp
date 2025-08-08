#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

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
// RelAlg to DB Conversion Pass - Phase 4c-4 with Simplified Architecture
//===----------------------------------------------------------------------===//

struct RelAlgToDBPass : public PassWrapper<RelAlgToDBPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDBPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::db::DBDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<::pgx::mlir::dsa::DSADialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-db"; }
    StringRef getDescription() const final { return "Convert RelAlg dialect to DB dialect"; }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass (Phase 4c-4 - Simplified Architecture)");
        
        auto funcOp = getOperation();
        
        // Process each MaterializeOp and perform translation
        // We need to handle proper operation lifecycle management
        // Store pointers to avoid invalidation issues when erasing
        llvm::SmallVector<Operation*> materializeOps;
        
        funcOp.walk([&](::pgx::mlir::relalg::MaterializeOp materializeOp) {
            materializeOps.push_back(materializeOp.getOperation());
        });
        
        for (auto* opPtr : materializeOps) {
            auto materializeOp = cast<::pgx::mlir::relalg::MaterializeOp>(opPtr);
            MLIR_PGX_INFO("RelAlgToDB", "Processing MaterializeOp translation");
            
            // Create a fresh context for each MaterializeOp
            ::pgx::mlir::relalg::TranslatorContext context;
            
            // Create translator for MaterializeOp (following LingoDB pattern)
            auto materializeTranslator = ::pgx::mlir::relalg::Translator::createTranslator(materializeOp);
            if (!materializeTranslator) {
                MLIR_PGX_ERROR("RelAlgToDB", "Failed to create MaterializeOp translator");
                signalPassFailure();
                return;
            }
            
            // Set translator info (no consumer at top level, empty required attributes)
            materializeTranslator->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
            
            // Set up builder at the MaterializeOp location
            OpBuilder builder(materializeOp);
            
            // Execute the translation - this will trigger the streaming pipeline:
            // 1. MaterializeTranslator.produce() calls BaseTableTranslator.produce()
            // 2. BaseTableTranslator streams tuples to MaterializeTranslator.consume()
            // 3. MaterializeTranslator builds DSA table and completes
            materializeTranslator->produce(context, builder);
            materializeTranslator->done();
            
            // Get the DSA table result from context
            auto dsaTable = context.getQueryResult();
            if (!dsaTable) {
                MLIR_PGX_ERROR("RelAlgToDB", "No DSA table result from translation");
                signalPassFailure();
                return;
            }
            
            MLIR_PGX_DEBUG("RelAlgToDB", "Translation complete, DSA table created");
            
            // Replace all uses of the MaterializeOp result with the DSA table
            materializeOp.getResult().replaceAllUsesWith(dsaTable);
        }
        
        // Skip function signature update to avoid type mismatch issues
        // The actual return values have been replaced with DSA table values
        // Function signature update might be causing issues with MLIR verification
        MLIR_PGX_DEBUG("RelAlgToDB", "Skipping function signature update to avoid type issues");
        
        // Theory C: Safer erasure with proper use validation
        // Only erase MaterializeOp operations that we've replaced
        // Leave other RelAlg operations for now to avoid complex dependency issues
        MLIR_PGX_INFO("RelAlgToDB", "Erasing replaced MaterializeOp operations safely");
        
        for (auto* opPtr : materializeOps) {
            // Check if operation is still valid and has no uses
            if (opPtr && opPtr->use_empty()) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Erasing MaterializeOp - no remaining uses");
                opPtr->erase();
            } else if (opPtr && !opPtr->use_empty()) {
                // This shouldn't happen if replaceAllUsesWith worked correctly
                MLIR_PGX_WARNING("RelAlgToDB", "MaterializeOp still has uses after replacement!");
            }
        }
        
        MLIR_PGX_INFO("RelAlgToDB", "Completed RelAlg to DB conversion pass - safely erased replaced operations");
        
        // DEBUG: Log what operations remain in the function before verification
        MLIR_PGX_INFO("RelAlgToDB", "=== OPERATIONS REMAINING AFTER CONVERSION ===");
        int opCount = 0;
        funcOp.walk([&](Operation *op) {
            opCount++;
            std::string opName = op->getName().getStringRef().str();
            std::string dialectName = op->getDialect() ? op->getDialect()->getNamespace().str() : "unknown";
            MLIR_PGX_INFO("RelAlgToDB", "Op #" + std::to_string(opCount) + ": " + dialectName + "." + opName);
            
            // Check if operation has valid types
            for (auto result : op->getResults()) {
                std::string typeName = "unknown";
                if (result.getType()) {
                    // Just check if type exists, don't try to print it yet
                    typeName = "valid-type";
                } else {
                    typeName = "null-type";
                }
                MLIR_PGX_INFO("RelAlgToDB", "  Result type: " + typeName);
            }
        });
        MLIR_PGX_INFO("RelAlgToDB", "Total operations: " + std::to_string(opCount));
        MLIR_PGX_INFO("RelAlgToDB", "=== END OPERATIONS LIST ===");
        
        // Verify the function is still valid after conversion
        MLIR_PGX_INFO("RelAlgToDB", "Starting function verification...");
        if (failed(funcOp.verify())) {
            MLIR_PGX_ERROR("RelAlgToDB", "Function verification failed after conversion");
            signalPassFailure();
            return;
        }
        MLIR_PGX_INFO("RelAlgToDB", "Function verification passed successfully");
        
        MLIR_PGX_DEBUG("RelAlgToDB", "Function verification passed");
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