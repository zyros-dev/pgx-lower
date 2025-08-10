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
// RelAlg to DB+DSA Conversion Pass - PostgreSQL SPI Integration Architecture
// 
// Phase 4d Implementation: Streaming producer-consumer pattern that generates
// mixed DB operations (for PostgreSQL table access via SPI) and DSA operations
// (for result materialization). Follows LingoDB Translator pattern for
// constant memory usage with one-tuple-at-a-time processing.
//===----------------------------------------------------------------------===//

struct RelAlgToDBPass : public PassWrapper<RelAlgToDBPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDBPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::mlir::db::DBDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<::pgx::mlir::dsa::DSADialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-db"; }
    StringRef getDescription() const final { 
        return "Convert RelAlg operations to mixed DB+DSA operations with PostgreSQL SPI integration"; 
    }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to mixed DB+DSA conversion (PostgreSQL SPI streaming pattern)");
        
        ::pgx::mlir::relalg::TranslatorContext loweringContext;
        SmallVector<::pgx::mlir::relalg::MaterializeOp> processedOps;
        SmallVector<::pgx::mlir::relalg::BaseTableOp> consumedBaseTableOps;
        
        getOperation().walk([&](Operation* op) {
            if (isTranslationHook(op)) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Processing translation hook for: " + op->getName().getStringRef().str());
                auto node = ::pgx::mlir::relalg::Translator::createTranslator(op);
                
                // Initialize translator with context for early column setup
                // This allows translators to initialize columns before produce() is called
                node->initializeWithContext(loweringContext);
                
                node->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
                OpBuilder builder(op);
                node->produce(loweringContext, builder);
                node->done();
                
                // Store MaterializeOp for later processing to avoid iterator invalidation
                if (auto materializeOp = dyn_cast<::pgx::mlir::relalg::MaterializeOp>(op)) {
                    processedOps.push_back(materializeOp);
                    
                    // Track consumed BaseTableOps
                    auto inputOp = materializeOp.getRel().getDefiningOp();
                    if (inputOp) {
                        if (auto baseTableOp = dyn_cast<::pgx::mlir::relalg::BaseTableOp>(inputOp)) {
                            consumedBaseTableOps.push_back(baseTableOp);
                        }
                    }
                }
            }
        });
        
        // Process MaterializeOps after walk to avoid iterator invalidation
        // With PostgreSQL SPI integration, MaterializeOp doesn't produce a DSA table
        // Instead, it streams results directly to PostgreSQL via StoreResultOp + StreamResultsOp
        for (auto materializeOp : processedOps) {
            // MaterializeOp has been fully replaced by DB/DSA operations that stream to PostgreSQL
            // No DSA table is stored in context - results go directly through SPI
            if (materializeOp->use_empty()) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Erasing MaterializeOp after PostgreSQL SPI streaming");
                materializeOp.erase();
            } else {
                // If MaterializeOp has uses, replace with a dummy value since results are streamed
                // This shouldn't happen in practice as MaterializeOp is terminal
                MLIR_PGX_WARNING("RelAlgToDB", "MaterializeOp has unexpected uses after SPI streaming");
                OpBuilder builder(materializeOp);
                // Create a dummy i1 constant as replacement
                auto dummyOp = builder.create<::mlir::arith::ConstantIntOp>(
                    materializeOp.getLoc(), 0, 1);
                materializeOp.replaceAllUsesWith(dummyOp.getResult());
                materializeOp.erase();
            }
        }
        
        updateFunctionSignature(getOperation());
        eraseConsumedRelAlgOperations(consumedBaseTableOps);
        
        MLIR_PGX_INFO("RelAlgToDB", "Pass execution completed - PostgreSQL SPI streaming architecture");
    }
    
private:
    bool isTranslationHook(Operation* op) {
        return isa<::pgx::mlir::relalg::MaterializeOp>(op);
    }
    
    // Keep existing proven method for updating function signatures
    void updateFunctionSignature(func::FuncOp funcOp) {
        MLIR_PGX_DEBUG("RelAlgToDB", "Updating function signature");
        
        // Find actual return types from return operations
        SmallVector<Type> actualReturnTypes;
        funcOp.walk([&](func::ReturnOp returnOp) {
            for (auto operand : returnOp.getOperands()) {
                actualReturnTypes.push_back(operand.getType());
            }
        });
        
        if (!actualReturnTypes.empty()) {
            auto funcType = funcOp.getFunctionType();
            // Only update if types actually changed
            bool needsUpdate = false;
            if (actualReturnTypes.size() == funcType.getNumResults()) {
                for (size_t i = 0; i < actualReturnTypes.size(); ++i) {
                    if (actualReturnTypes[i] != funcType.getResult(i)) {
                        needsUpdate = true;
                        break;
                    }
                }
            } else {
                needsUpdate = true;
            }
            
            if (needsUpdate) {
                auto newFuncType = FunctionType::get(funcOp.getContext(),
                                                    funcType.getInputs(),
                                                    actualReturnTypes);
                funcOp.setType(newFuncType);
                MLIR_PGX_INFO("RelAlgToDB", "Updated function signature to use DSA table types");
            }
        }
    }
    
    // Safe erasure of consumed RelAlg operations (only those processed by MaterializeOps)
    void eraseConsumedRelAlgOperations(const SmallVector<::pgx::mlir::relalg::BaseTableOp>& consumedBaseTableOps) {
        MLIR_PGX_DEBUG("RelAlgToDB", "Erasing consumed RelAlg operations");
        
        // Collect RelAlg operations to erase
        SmallVector<Operation*> opsToErase;
        // Only erase BaseTableOps that were actually consumed by MaterializeOps
        for (auto baseTableOp : consumedBaseTableOps) {
            if (baseTableOp->use_empty()) {
                opsToErase.push_back(baseTableOp);
                MLIR_PGX_DEBUG("RelAlgToDB", "Erasing consumed BaseTableOp");
            } else {
                MLIR_PGX_DEBUG("RelAlgToDB", "Consumed BaseTableOp still has uses, not erasing");
            }
        }
        
        // Erase in reverse order to handle dependencies correctly
        for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
            MLIR_PGX_DEBUG("RelAlgToDB", "Erasing operation: " + 
                          (*it)->getName().getStringRef().str());
            (*it)->erase();
        }
        
        MLIR_PGX_INFO("RelAlgToDB", "Erased " + std::to_string(opsToErase.size()) + " consumed RelAlg operations");
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