#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pgx {
namespace mlir {
namespace relalg {

class MaterializeTranslator : public Translator {
private:
    ::pgx::mlir::relalg::MaterializeOp materializeOp;
    OrderedAttributes orderedAttributes;
    
public:
    explicit MaterializeTranslator(::pgx::mlir::relalg::MaterializeOp op) 
        : Translator(op), materializeOp(op) {
        MLIR_PGX_DEBUG("RelAlg", "Created MaterializeTranslator");
        
        // Create child translator for the input relation
        auto inputOp = op.getRel().getDefiningOp();
        if (inputOp) {
            addChild(createTranslator(inputOp));
        }
    }
    
    ColumnSet getAvailableColumns() override {
        // MaterializeOp doesn't produce columns for consumption - it materializes to a table
        return ColumnSet();
    }
    
    void setInfo(Translator* consumer, const ColumnSet& requiredAttributes) override {
        Translator::setInfo(consumer, requiredAttributes);
        
        // Don't create our own columns - we'll use the columns from the child
        // The child will tell us what columns are available
        if (!children.empty()) {
            // Pass empty required attributes to child to get all available columns
            children[0]->setInfo(this, ColumnSet());
            
            // Get available columns from child and use them
            auto childColumns = children[0]->getAvailableColumns();
            
            // For Test 1, we expect one column (id)
            // Add all available columns to our OrderedAttributes
            for (const Column* col : childColumns) {
                orderedAttributes.insert(col);
            }
            
            MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator configured with " + 
                          std::to_string(orderedAttributes.getAttrs().size()) + " columns from child");
        }
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        MLIR_PGX_INFO("RelAlg", "MaterializeTranslator::produce() - Starting materialization with DB operations");
        
        auto loc = materializeOp.getLoc();
        
        // Process input relation - this will call consume() for each tuple
        processInputRelation(context, builder);
        
        // Stream the results after all tuples are processed
        streamResults(builder, loc);
    }
    
private:
    
    // Process input relation by calling child's produce
    void processInputRelation(TranslatorContext& context, ::mlir::OpBuilder& builder) {
        if (!children.empty()) {
            children[0]->setConsumer(this);
            children[0]->produce(context, builder);
            children[0]->done();
        }
    }
    
    // Stream the results after all tuples are processed
    void streamResults(::mlir::OpBuilder& builder, ::mlir::Location loc) {
        // Generate db.stream_results to output the accumulated results
        builder.create<::pgx::db::StreamResultsOp>(loc);
        
        MLIR_PGX_DEBUG("RelAlg", "Generated db.stream_results operation");
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::consume() - Processing tuple from child using DB operations");
        
        auto loc = materializeOp.getLoc();
        
        // Process each column and store in result tuple
        for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            auto val = orderedAttributes.resolve(context, i);
            
            // Convert to nullable type for DB operation compatibility
            auto nullableVal = builder.create<::pgx::db::AsNullableOp>(
                loc, ::pgx::db::NullableI64Type::get(builder.getContext()), val);
            
            // Store the value in the result tuple at the appropriate field index
            auto fieldIndex = builder.create<::mlir::arith::ConstantIndexOp>(loc, i);
            builder.create<::pgx::db::StoreResultOp>(
                loc, nullableVal, fieldIndex);
            
            MLIR_PGX_DEBUG("RelAlg", "Stored field " + std::to_string(i) + " using db.store_result");
        }
        
        MLIR_PGX_DEBUG("RelAlg", "Completed storing tuple fields");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::done() - Completed DB-based materialization");
        
        // The MaterializeOp has been fully lowered to DB operations
        materializeOp.erase();
    }
};

// Factory function
std::unique_ptr<Translator> createMaterializeTranslator(::mlir::Operation* op) {
    auto materializeOp = ::mlir::dyn_cast<::pgx::mlir::relalg::MaterializeOp>(op);
    if (!materializeOp) {
        MLIR_PGX_ERROR("RelAlg", "createMaterializeTranslator called with non-MaterializeOp");
        return createDummyTranslator(op);
    }
    return std::make_unique<MaterializeTranslator>(materializeOp);
}

} // namespace relalg
} // namespace mlir
} // namespace pgx