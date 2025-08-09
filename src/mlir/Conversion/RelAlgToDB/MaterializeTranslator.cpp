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
    ::mlir::Value tableBuilder;  // DSA table builder for PostgreSQL result materialization
    ::mlir::Value finalTable;    // The finalized DSA table (PostgreSQL query result)
    
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
        MLIR_PGX_INFO("RelAlg", "MaterializeTranslator::produce() - Starting PostgreSQL result materialization with DSA operations");
        
        // Log builder state before operations
        auto* currentBlock = builder.getInsertionBlock();
        if (currentBlock) {
            MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator - insertion block has " + 
                          std::to_string(std::distance(currentBlock->begin(), currentBlock->end())) + " ops");
            if (currentBlock->getTerminator()) {
                MLIR_PGX_DEBUG("RelAlg", "Block has terminator: " + 
                              currentBlock->getTerminator()->getName().getStringRef().str());
            } else {
                MLIR_PGX_WARNING("RelAlg", "Block has NO terminator before MaterializeTranslator operations");
            }
        }
        
        auto loc = materializeOp.getLoc();
        
        // Create DSA table builder for PostgreSQL result collection (Phase 4d)
        // DSA operations handle result materialization, DB operations handle PostgreSQL access
        auto tupleType = createTupleType(builder);
        auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(builder.getContext(), tupleType);
        tableBuilder = builder.create<::pgx::mlir::dsa::CreateDSOp>(loc, tableBuilderType);
        
        MLIR_PGX_DEBUG("RelAlg", "Created DSA table builder for PostgreSQL result materialization");
        
        // Process input relation - triggers streaming consume() calls for each PostgreSQL tuple
        processInputRelation(context, builder);
        
        // Finalize DSA table (PostgreSQL result ready for return)
        finalizeAndStreamResults(builder, loc, context);
        
        // Log builder state after operations
        if (currentBlock) {
            MLIR_PGX_DEBUG("RelAlg", "After MaterializeTranslator ops - block has " + 
                          std::to_string(std::distance(currentBlock->begin(), currentBlock->end())) + " ops");
            if (currentBlock->getTerminator()) {
                MLIR_PGX_DEBUG("RelAlg", "Block still has terminator after MaterializeTranslator ops");
            } else {
                MLIR_PGX_ERROR("RelAlg", "Block LOST terminator during MaterializeTranslator operations!");
            }
        }
    }
    
private:
    
    // Create PostgreSQL result tuple type from ordered attributes
    ::mlir::TupleType createTupleType(::mlir::OpBuilder& builder) {
        // Build the tuple type from PostgreSQL table column types
        // For Test 1: single column (id: i32) from PostgreSQL table
        return orderedAttributes.getTupleType(builder.getContext());
    }
    
    // Process PostgreSQL input relation via streaming producer-consumer pattern
    void processInputRelation(TranslatorContext& context, ::mlir::OpBuilder& builder) {
        if (!children.empty()) {
            children[0]->setConsumer(this);  // Set this as consumer for streaming
            children[0]->produce(context, builder);  // Triggers PostgreSQL tuple iteration
            children[0]->done();
        }
    }
    
    // Finalize PostgreSQL result DSA table (Phase 4d architecture)
    void finalizeAndStreamResults(::mlir::OpBuilder& builder, ::mlir::Location loc, 
                                  TranslatorContext& context) {
        // Finalize DSA table builder - creates PostgreSQL query result table
        auto tupleType = createTupleType(builder);
        auto tableType = ::pgx::mlir::dsa::TableType::get(builder.getContext(), tupleType);
        auto table = builder.create<::pgx::mlir::dsa::FinalizeOp>(loc, tableType, tableBuilder);
        
        MLIR_PGX_DEBUG("RelAlg", "Finalized DSA table containing PostgreSQL query results");
        
        // Store PostgreSQL result table for pass value replacement
        finalTable = table.getResult();
        
        // Store in context for RelAlgToDB pass to retrieve and replace MaterializeOp uses
        context.setQueryResult(finalTable);
        
        // Pass handles MaterializeOp → DSA table replacement (no db.stream_results needed)
        MLIR_PGX_DEBUG("RelAlg", "PostgreSQL DSA result table finalized and stored for pass replacement");
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::consume() - Processing single PostgreSQL tuple using DSA operations");
        
        auto loc = materializeOp.getLoc();
        
        // Collect PostgreSQL field values for current tuple materialization
        llvm::SmallVector<::mlir::Value> tupleValues;
        for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            const Column* col = orderedAttributes.getAttrs()[i];
            MLIR_PGX_DEBUG("RelAlg", "Resolving PostgreSQL column " + std::to_string(i));
            
            auto val = orderedAttributes.resolve(context, i);
            tupleValues.push_back(val);
            
            MLIR_PGX_DEBUG("RelAlg", "Collected PostgreSQL field " + std::to_string(i) + " for DSA materialization");
        }
        
        // Append PostgreSQL field values to DSA result table (one tuple at a time)
        builder.create<::pgx::mlir::dsa::DSAppendOp>(loc, tableBuilder, tupleValues);
        
        MLIR_PGX_DEBUG("RelAlg", "Appended PostgreSQL tuple values to DSA result builder");
        
        // Finalize current row in result builder (streaming pattern)
        builder.create<::pgx::mlir::dsa::NextRowOp>(loc, tableBuilder);
        
        MLIR_PGX_DEBUG("RelAlg", "Finalized PostgreSQL row in DSA builder - ready for next streaming tuple");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::done() - Completed PostgreSQL result materialization with DSA");
        
        // Pass handles MaterializeOp cleanup after DSA table replacement (Phase 4d pattern)
        // The pass calls replaceAllUsesWith(DSA table) then erases MaterializeOp
    }
    
    // Get the finalized DSA table value
    ::mlir::Value getFinalTable() const {
        return finalTable;
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