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
        
        // Process each MaterializeOp using LingoDB pattern
        for (auto materializeOp : materializeOps) {
            ::pgx::mlir::relalg::TranslatorContext loweringContext;
            
            // Create translator and execute
            auto node = ::pgx::mlir::relalg::Translator::createTranslator(materializeOp);
            node->setInfo(nullptr, ::pgx::mlir::relalg::ColumnSet());
            
            // CRITICAL: Set insertion point BEFORE the materialize op
            // This ensures generated operations go in the right place
            OpBuilder builder(materializeOp);
            builder.setInsertionPoint(materializeOp);
            
            node->produce(loweringContext, builder);
            node->done();
            
            // Get the DSA table result from context and replace MaterializeOp
            auto dsaTable = loweringContext.getQueryResult();
            if (dsaTable) {
                materializeOp.replaceAllUsesWith(dsaTable);
                materializeOp.erase();
            }
        }
        
        // Keep the proven function signature update
        updateFunctionSignature(getOperation());
        
        // Erase remaining RelAlg operations safely  
        eraseRelAlgOperations();
        
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
        MLIR_PGX_DEBUG("RelAlgToDB", "Erasing RelAlg operations");
        
        // Collect RelAlg operations to erase (in reverse order)
        SmallVector<Operation*> opsToErase;
        getOperation().walk([&](Operation* op) {
            if (op->getDialect() && 
                op->getDialect()->getNamespace() == "relalg") {
                // Only erase if the operation has no uses (should have been replaced)
                if (op->use_empty()) {
                    opsToErase.push_back(op);
                }
            }
        });
        
        // Erase in reverse order to handle dependencies correctly
        for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
            (*it)->erase();
        }
        
        MLIR_PGX_INFO("RelAlgToDB", "Erased " + std::to_string(opsToErase.size()) + " RelAlg operations");
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