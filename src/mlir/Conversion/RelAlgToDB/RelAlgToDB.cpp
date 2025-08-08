#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// RelAlg to DB Conversion Pass - Using LingoDB Translator Pattern
//===----------------------------------------------------------------------===//

struct RelAlgToDBPass : public PassWrapper<RelAlgToDBPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDBPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::db::DBDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<::pgx::mlir::dsa::DSADialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-db"; }
    StringRef getDescription() const final { return "Convert RelAlg dialect to DB dialect"; }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass (LingoDB Pattern)");
        
        ::pgx::mlir::relalg::TranslatorContext loweringContext;
        SmallVector<::pgx::mlir::relalg::MaterializeOp> processedOps;
        
        getOperation().walk([&](Operation* op) {
            if (isTranslationHook(op)) {
                auto node = ::pgx::mlir::relalg::Translator::createTranslator(op);
                node->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
                OpBuilder builder(op);
                node->produce(loweringContext, builder);
                node->done();
                
                // Store MaterializeOp for later processing to avoid iterator invalidation
                if (auto materializeOp = dyn_cast<::pgx::mlir::relalg::MaterializeOp>(op)) {
                    processedOps.push_back(materializeOp);
                }
            }
        });
        
        // Process MaterializeOps after walk to avoid iterator invalidation
        for (auto materializeOp : processedOps) {
            auto dsaTable = loweringContext.getQueryResult();
            if (dsaTable) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Replacing MaterializeOp with DSA table");
                materializeOp.replaceAllUsesWith(dsaTable);
                materializeOp.erase();
            }
        }
        
        updateFunctionSignature(getOperation());
        eraseRemainingRelAlgOperations();
        
        MLIR_PGX_INFO("RelAlgToDB", "Pass execution completed - LingoDB pattern");
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
    
    // Safe erasure of remaining RelAlg operations (BaseTable, etc) after translation
    void eraseRemainingRelAlgOperations() {
        MLIR_PGX_DEBUG("RelAlgToDB", "Erasing consumed RelAlg operations");
        
        // Collect RelAlg operations to erase
        SmallVector<Operation*> opsToErase;
        getOperation().walk([&](Operation* op) {
            // CRITICAL: Never erase func.return operations!
            if (isa<func::ReturnOp>(op)) {
                return;  // Skip return operations
            }
            
            if (op->getDialect() && 
                op->getDialect()->getNamespace() == "relalg") {
                // Only erase BaseTableOps and other RelAlg ops (NOT MaterializeOps - already handled)
                if (isa<::pgx::mlir::relalg::MaterializeOp>(op)) {
                    // Skip MaterializeOps - already handled in main loop
                    return;
                } else if (isa<::pgx::mlir::relalg::BaseTableOp>(op)) {
                    // BaseTableOps should have no uses after MaterializeOps are replaced
                    if (op->use_empty()) {
                        opsToErase.push_back(op);
                    } else {
                        MLIR_PGX_DEBUG("RelAlgToDB", "BaseTableOp still has uses, not erasing");
                    }
                }
                // Other RelAlg operations can be erased if they have no uses
                else if (op->use_empty()) {
                    opsToErase.push_back(op);
                }
            }
        });
        
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