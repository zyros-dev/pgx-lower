#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "llvm/Support/ErrorHandling.h"

namespace pgx {
namespace mlir {
namespace relalg {

// Factory method to create appropriate translator based on operation type
std::unique_ptr<Translator> Translator::createTranslator(::mlir::Operation* op) {
    MLIR_PGX_INFO("RelAlg", "Creating translator for operation: " + op->getName().getStringRef().str());
    
    // Use TypeSwitch to dispatch to appropriate translator factory
    return ::llvm::TypeSwitch<::mlir::Operation*, std::unique_ptr<Translator>>(op)
        .Case<::pgx::mlir::relalg::BaseTableOp>([&](auto x) { 
            return createBaseTableTranslator(x); 
        })
        .Case<::pgx::mlir::relalg::MaterializeOp>([&](auto x) { 
            return createMaterializeTranslator(x); 
        })
        .Default([&](auto x) {
            // For now, use dummy translator for unimplemented operations
            MLIR_PGX_WARNING("RelAlg", "No specific translator for operation: " + x->getName().getStringRef().str() + ", using DummyTranslator");
            return createDummyTranslator(x);
        });
}

// Dummy translator for testing infrastructure
class DummyTranslator : public Translator {
public:
    explicit DummyTranslator(::mlir::Operation* op) : Translator(op) {
        MLIR_PGX_DEBUG("RelAlg", "Created DummyTranslator for operation: " + op->getName().getStringRef().str());
    }
    
    ColumnSet getAvailableColumns() override {
        // Dummy translator doesn't produce any columns
        return ColumnSet();
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        MLIR_PGX_DEBUG("RelAlg", "DummyTranslator::produce() called");
        
        // For operations with children, process them
        if (!children.empty()) {
            produceChildren(context, builder);
        }
        
        // For leaf operations or after children are processed
        if (consumer) {
            consumer->consume(this, builder, context);
        }
    }
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) override {
        MLIR_PGX_DEBUG("RelAlg", "DummyTranslator::consume() called from child");
        
        // Pass through to parent consumer if exists
        if (consumer) {
            consumer->consume(this, builder, context);
        }
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "DummyTranslator::done() called");
        // No-op for dummy translator
    }
};

// Factory function for DummyTranslator
std::unique_ptr<Translator> createDummyTranslator(::mlir::Operation* op) {
    return std::make_unique<DummyTranslator>(op);
}

} // namespace relalg
} // namespace mlir
} // namespace pgx