#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H

#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include <memory>
#include <vector>

namespace pgx {
namespace mlir {
namespace relalg {

// Abstract base class for RelAlg operation translators (Phase 4d)
// Implements LingoDB produce/consume pattern for PostgreSQL streaming tuple processing
// Enables constant memory usage with one-tuple-at-a-time PostgreSQL SPI integration
class Translator {
protected:
    // Child translators (for operations with relational inputs)
    std::vector<std::unique_ptr<Translator>> children;
    
    // Parent translator that will consume this translator's output
    Translator* consumer = nullptr;
    
    // The RelAlg operation being translated
    ::mlir::Operation* op = nullptr;
    
    // Required columns from parent operation
    ColumnSet requiredAttributes;

public:
    // Constructors
    Translator() = default;
    explicit Translator(::mlir::Operation* op) : op(op) {}
    virtual ~Translator() = default;
    
    // Core translation methods
    
    // initializeWithContext() allows early access to TranslatorContext for column setup
    // This is called before setInfo() for translators that need early column initialization
    virtual void initializeWithContext(TranslatorContext& context) {
        // Default implementation does nothing
        // Override in translators that need early column initialization (e.g., BaseTableTranslator)
    }
    
    // setInfo() is called before produce to set consumer and required columns
    virtual void setInfo(Translator* consumer, const ColumnSet& requiredAttributes) {
        this->consumer = consumer;
        this->requiredAttributes = requiredAttributes;
        propagateInfo();
    }
    
    // getAvailableColumns() returns the columns this operation produces
    virtual ColumnSet getAvailableColumns() = 0;
    
    // produce() initiates PostgreSQL table access and streaming tuple processing
    virtual void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
    
    // consume() processes individual PostgreSQL tuples from child translators (streaming pattern)
    virtual void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) = 0;
    
    // done() performs any cleanup after translation is complete
    virtual void done() {}
    
    // Child management
    void addChild(std::unique_ptr<Translator> child) {
        children.push_back(std::move(child));
    }
    
    void setConsumer(Translator* parent) {
        consumer = parent;
    }
    
    Translator* getConsumer() const {
        return consumer;
    }
    
    // Factory method to create PostgreSQL-compatible translator for RelAlg operation
    static std::unique_ptr<Translator> createTranslator(::mlir::Operation* op);
    
    // Operation access
    ::mlir::Operation* getOperation() const { return op; }
    
    // Helper method to call produce on all children
    void produceChildren(TranslatorContext& context, ::mlir::OpBuilder& builder) {
        for (auto& child : children) {
            child->setConsumer(this);
            child->produce(context, builder);
        }
    }
    
protected:
    // Propagate column requirements to child translators
    void propagateInfo() {
        // Base implementation - can be overridden by specific translators
        for (auto& child : children) {
            if (child) {
                child->setInfo(this, requiredAttributes);
            }
        }
    }
    
    // Helper method to merge relational blocks
    std::vector<::mlir::Value> mergeRelationalBlock(::mlir::Block* dest, ::mlir::Operation* op, 
                                                    ::mlir::function_ref<::mlir::Block*(::mlir::Operation*)> getBlockFn, 
                                                    TranslatorContext& context, 
                                                    TranslatorContext::AttributeResolverScope& scope);
};

// Factory functions for specific translators (to be implemented)
std::unique_ptr<Translator> createBaseTableTranslator(::mlir::Operation* op);
std::unique_ptr<Translator> createMaterializeTranslator(::mlir::Operation* op);
std::unique_ptr<Translator> createDummyTranslator(::mlir::Operation* op);
std::unique_ptr<Translator> createMapTranslator(::mlir::Operation* op);
std::unique_ptr<Translator> createJoinTranslator(::mlir::Operation* op);

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H