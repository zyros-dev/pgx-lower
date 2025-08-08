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
        
        // Collect MaterializeOps to process
        SmallVector<::pgx::mlir::relalg::MaterializeOp> materializeOps;
        getOperation().walk([&](::pgx::mlir::relalg::MaterializeOp op) {
            materializeOps.push_back(op);
        });
        
        // Handle empty functions gracefully
        if (materializeOps.empty()) {
            MLIR_PGX_INFO("RelAlgToDB", "No MaterializeOps found - pass completes successfully");
            return;
        }
        
        // Process each MaterializeOp using LingoDB pattern
        for (auto materializeOp : materializeOps) {
            ::pgx::mlir::relalg::TranslatorContext loweringContext;
            
            // Create translator and execute
            auto node = ::pgx::mlir::relalg::Translator::createTranslator(materializeOp);
            node->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
            
            // CRITICAL: Create builder with proper context
            // Use the function's builder context to maintain consistency
            OpBuilder builder(&getContext());
            
            // Ensure we have a valid insertion block
            if (!materializeOp->getBlock()) {
                MLIR_PGX_ERROR("RelAlgToDB", "MaterializeOp has no parent block!");
                continue;
            }
            
            // Set insertion point BEFORE the materialize op
            // This ensures generated operations go in the right place
            builder.setInsertionPoint(materializeOp);
            
            // EXTRA SAFETY: Ensure we're not inserting after a terminator
            if (auto* block = builder.getInsertionBlock()) {
                if (block->empty()) {
                    MLIR_PGX_ERROR("RelAlgToDB", "Insertion block is empty!");
                    continue;
                }
                // Ensure we're inserting before any terminator
                if (auto* terminator = block->getTerminator()) {
                    if (builder.getInsertionPoint() == block->end() ||
                        &*builder.getInsertionPoint() == terminator) {
                        // We're at or after the terminator, move before it
                        builder.setInsertionPoint(terminator);
                    }
                }
            }
            
            // Verify block structure before produce
            if (auto* block = builder.getInsertionBlock()) {
                if (!block->getTerminator()) {
                    MLIR_PGX_ERROR("RelAlgToDB", "Block has no terminator before produce!");
                } else {
                    MLIR_PGX_DEBUG("RelAlgToDB", "Block has terminator before produce");
                }
            }
            
            node->produce(loweringContext, builder);
            node->done();
            
            // Verify block structure after produce
            if (auto* block = builder.getInsertionBlock()) {
                if (!block->getTerminator()) {
                    MLIR_PGX_ERROR("RelAlgToDB", "Block has no terminator after produce!");
                    // This is a critical error - the produce() method broke the block structure
                } else {
                    MLIR_PGX_DEBUG("RelAlgToDB", "Block still has terminator after produce");
                }
            }
            
            // Get the DSA table result from context and replace MaterializeOp
            auto dsaTable = loweringContext.getQueryResult();
            if (dsaTable) {
                MLIR_PGX_DEBUG("RelAlgToDB", "Replacing MaterializeOp uses with DSA table");
                
                // CRITICAL: Replace all uses of MaterializeOp with DSA table
                // This includes return operations
                if (!materializeOp.use_empty()) {
                    MLIR_PGX_DEBUG("RelAlgToDB", "MaterializeOp has uses, replacing with DSA table");
                    
                    // Before replacing, check if any uses are return operations
                    for (auto& use : materializeOp->getUses()) {
                        if (auto returnOp = dyn_cast<func::ReturnOp>(use.getOwner())) {
                            MLIR_PGX_DEBUG("RelAlgToDB", "Found return operation using MaterializeOp");
                        }
                    }
                    
                    materializeOp.replaceAllUsesWith(dsaTable);
                    
                    // Verify return operations are updated
                    getOperation().walk([&](func::ReturnOp returnOp) {
                        for (auto operand : returnOp.getOperands()) {
                            if (operand == dsaTable) {
                                MLIR_PGX_DEBUG("RelAlgToDB", "Return operation successfully updated with DSA table");
                            }
                        }
                    });
                }
                
                MLIR_PGX_DEBUG("RelAlgToDB", "Erasing MaterializeOp after replacement");
                materializeOp.erase();
            } else {
                MLIR_PGX_ERROR("RelAlgToDB", "No DSA table result from translator!");
            }
        }
        
        // CRITICAL: Before erasing RelAlg operations, ensure all terminators are intact
        // This includes verifying that return operations still exist and are valid
        getOperation().walk([&](func::ReturnOp returnOp) {
            if (returnOp.getNumOperands() == 0) {
                MLIR_PGX_ERROR("RelAlgToDB", "Found return operation with no operands!");
            }
        });
        
        // Keep the proven function signature update
        updateFunctionSignature(getOperation());
        
        // Erase remaining RelAlg operations safely  
        eraseRelAlgOperations();
        
        // Final verification: ensure basic block structure is maintained
        bool hasStructuralIssues = false;
        getOperation().walk([&](Block* block) {
            if (!block->getTerminator() && !block->empty()) {
                MLIR_PGX_ERROR("RelAlgToDB", "Found block without terminator!");
                hasStructuralIssues = true;
                
                // Log the last operation in the block for debugging
                if (!block->empty()) {
                    auto& lastOp = block->back();
                    MLIR_PGX_ERROR("RelAlgToDB", "Last operation in block: " + 
                                  lastOp.getName().getStringRef().str());
                }
            }
        });
        
        if (hasStructuralIssues) {
            MLIR_PGX_ERROR("RelAlgToDB", "Generated IR has structural issues - blocks without terminators");
            signalPassFailure();
            return;
        }
        
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
    
    // Safe erasure of RelAlg operations after translation
    void eraseRelAlgOperations() {
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
                // Only erase MaterializeOps and BaseTableOps that were consumed
                if (isa<::pgx::mlir::relalg::MaterializeOp>(op)) {
                    // MaterializeOps should have been replaced by DSA tables
                    if (op->use_empty()) {
                        opsToErase.push_back(op);
                    }
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