#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pgx {
namespace mlir {
namespace relalg {

class MaterializeTranslator : public Translator {
private:
    ::pgx::mlir::relalg::MaterializeOp materializeOp;
    OrderedAttributes orderedAttributes;
    ::mlir::Value tableBuilder;  // DSA table builder for result materialization
    
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
        MLIR_PGX_INFO("RelAlg", "MaterializeTranslator::produce() - Starting materialization with DSA operations");
        
        auto loc = materializeOp.getLoc();
        
        // Create DSA table builder based on column types
        auto tupleType = createTupleType(builder);
        auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(builder.getContext(), tupleType);
        tableBuilder = builder.create<::pgx::mlir::dsa::CreateDSOp>(loc, tableBuilderType);
        
        MLIR_PGX_DEBUG("RelAlg", "Created DSA table builder with tuple type");
        
        // Process input relation - this will call consume() for each tuple
        processInputRelation(context, builder);
        
        // Finalize the DSA table and stream results
        finalizeAndStreamResults(builder, loc);
    }
    
private:
    
    // Create tuple type from ordered attributes
    ::mlir::TupleType createTupleType(::mlir::OpBuilder& builder) {
        // Build the tuple type from ordered attribute types
        // For Test 1, we expect one column (id: i32)
        return orderedAttributes.getTupleType(builder.getContext());
    }
    
    // Process input relation by calling child's produce
    void processInputRelation(TranslatorContext& context, ::mlir::OpBuilder& builder) {
        if (!children.empty()) {
            children[0]->setConsumer(this);
            children[0]->produce(context, builder);
            children[0]->done();
        }
    }
    
    // Finalize the DSA table and stream results
    void finalizeAndStreamResults(::mlir::OpBuilder& builder, ::mlir::Location loc) {
        // Finalize the DSA table builder to create the final table
        auto tableType = ::pgx::mlir::dsa::TableType::get(builder.getContext());
        auto table = builder.create<::pgx::mlir::dsa::FinalizeOp>(loc, tableType, tableBuilder);
        
        MLIR_PGX_DEBUG("RelAlg", "Finalized DSA table");
        
        // Generate db.stream_results to output the materialized table
        builder.create<::pgx::db::StreamResultsOp>(loc);
        
        MLIR_PGX_DEBUG("RelAlg", "Generated db.stream_results operation");
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::consume() - Processing single tuple from child using DSA operations");
        
        auto loc = materializeOp.getLoc();
        
        // Collect values for all columns in this tuple
        llvm::SmallVector<::mlir::Value> tupleValues;
        for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            auto val = orderedAttributes.resolve(context, i);
            tupleValues.push_back(val);
            
            MLIR_PGX_DEBUG("RelAlg", "Collected field " + std::to_string(i) + " for DSA append");
        }
        
        // Append all values to the DSA table builder in one operation
        builder.create<::pgx::mlir::dsa::DSAppendOp>(loc, tableBuilder, tupleValues);
        
        MLIR_PGX_DEBUG("RelAlg", "Appended tuple values to DSA table builder");
        
        // Finalize the current row in the builder
        builder.create<::pgx::mlir::dsa::NextRowOp>(loc, tableBuilder);
        
        MLIR_PGX_DEBUG("RelAlg", "Finalized row in DSA table builder - ready for next tuple");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::done() - Completed DSA-based materialization");
        
        // The MaterializeOp has been fully lowered to DB+DSA operations
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