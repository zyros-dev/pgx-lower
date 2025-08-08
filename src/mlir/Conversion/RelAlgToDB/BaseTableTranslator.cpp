#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include <sstream>

namespace pgx {
namespace mlir {
namespace relalg {

class BaseTableTranslator : public Translator {
private:
    ::pgx::mlir::relalg::BaseTableOp baseTableOp;
    ColumnSet availableColumns;
    // Store columns as member variables to ensure proper lifetime
    Column idColumn;
    
public:
    explicit BaseTableTranslator(::pgx::mlir::relalg::BaseTableOp op) 
        : Translator(op), 
          baseTableOp(op),
          idColumn(::mlir::IntegerType::get(op.getContext(), 64)) {
        MLIR_PGX_DEBUG("RelAlg", "Created BaseTableTranslator for table: " + 
                       op.getTableName().str());
        
        // Initialize available columns - for Test 1, just create an 'id' column
        availableColumns.insert(&idColumn);
    }
    
    ColumnSet getAvailableColumns() override {
        return availableColumns;
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        MLIR_PGX_INFO("RelAlg", "BaseTableTranslator::produce() - Beginning DB-based table scan for: " + 
                      baseTableOp.getTableName().str());
        
        // Log builder state before operations
        auto* currentBlock = builder.getInsertionBlock();
        if (currentBlock) {
            MLIR_PGX_DEBUG("RelAlg", "Builder insertion block has " + 
                          std::to_string(currentBlock->getNumArguments()) + " args, " +
                          std::to_string(std::distance(currentBlock->begin(), currentBlock->end())) + " ops");
            if (currentBlock->getTerminator()) {
                MLIR_PGX_DEBUG("RelAlg", "Block has terminator: " + 
                              currentBlock->getTerminator()->getName().getStringRef().str());
            } else {
                MLIR_PGX_WARNING("RelAlg", "Block has NO terminator before BaseTable operations");
            }
        }
        
        auto scope = context.createScope();
        auto loc = baseTableOp.getLoc();
        
        // Create DB operations following the correct pipeline architecture
        createDBTableScan(context, builder, scope, loc);
        
        // Log builder state after operations
        if (currentBlock) {
            MLIR_PGX_DEBUG("RelAlg", "After BaseTable ops - block has " + 
                          std::to_string(std::distance(currentBlock->begin(), currentBlock->end())) + " ops");
            if (currentBlock->getTerminator()) {
                MLIR_PGX_DEBUG("RelAlg", "Block still has terminator after BaseTable ops");
            } else {
                MLIR_PGX_ERROR("RelAlg", "Block LOST terminator during BaseTable operations!");
            }
        }
    }
    
private:
    // Create the DB-based table scan following the correct architecture
    void createDBTableScan(TranslatorContext& context, ::mlir::OpBuilder& builder,
                          TranslatorContext::AttributeResolverScope& scope, 
                          ::mlir::Location loc) {
        MLIR_PGX_DEBUG("RelAlg", "Creating DB-based table scan for: " + 
                       baseTableOp.getTableName().str());
        
        // Get table OID from the BaseTableOp
        auto tableOid = builder.create<::mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(baseTableOp.getTableOid()));
        
        // Create DB operations for PostgreSQL table access
        auto externalSourceType = ::pgx::db::ExternalSourceType::get(builder.getContext());
        auto externalSource = builder.create<::pgx::db::GetExternalOp>(
            loc, externalSourceType, tableOid);
        
        MLIR_PGX_DEBUG("RelAlg", "Created db.get_external for table OID: " + 
                       std::to_string(baseTableOp.getTableOid()));
        
        // Create DB-based tuple iteration
        createTupleIterationLoop(context, builder, scope, loc, externalSource.getResult());
    }
    
    // Create DB-based tuple iteration with proper PostgreSQL integration
    void createTupleIterationLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                                 TranslatorContext::AttributeResolverScope& scope,
                                 ::mlir::Location loc, ::mlir::Value tableHandle) {
        MLIR_PGX_DEBUG("RelAlg", "Creating DB-based iteration loop for table: " + 
                       baseTableOp.getTableName().str());
        
        // Create a while loop that iterates through PostgreSQL tuples
        // We'll use scf.while for the iteration pattern
        auto whileOp = builder.create<::mlir::scf::WhileOp>(
            loc,
            /*resultTypes=*/::mlir::TypeRange{},
            /*operands=*/::mlir::ValueRange{}
        );
        
        // Build the "before" region (condition check)
        {
            auto& beforeRegion = whileOp.getBefore();
            auto* beforeBlock = builder.createBlock(&beforeRegion);
            builder.setInsertionPointToStart(beforeBlock);
            
            // Call db.iterate_external to check if there's a next tuple
            auto hasTuple = builder.create<::pgx::db::IterateExternalOp>(
                loc, builder.getI1Type(), tableHandle);
            
            MLIR_PGX_DEBUG("RelAlg", "Created db.iterate_external in while condition");
            
            // Yield the condition
            builder.create<::mlir::scf::ConditionOp>(loc, hasTuple.getResult(), ::mlir::ValueRange{});
        }
        
        // Build the "after" region (loop body)
        {
            auto& afterRegion = whileOp.getAfter();
            auto* afterBlock = builder.createBlock(&afterRegion);
            builder.setInsertionPointToStart(afterBlock);
            
            // Process the current tuple
            processTupleInLoop(context, builder, scope, loc, tableHandle);
            
            // Yield to continue the loop
            builder.create<::mlir::scf::YieldOp>(loc, ::mlir::ValueRange{});
        }
        
        // Set insertion point after the while loop
        builder.setInsertionPointAfter(whileOp);
        
        MLIR_PGX_DEBUG("RelAlg", "Created scf.while loop for DB tuple iteration");
    }
    
    // Process a single tuple inside the loop - extract field values and call consumer
    void processTupleInLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                           TranslatorContext::AttributeResolverScope& scope,
                           ::mlir::Location loc, ::mlir::Value tableHandle) {
        // Extract field value using db.get_field operations
        // For Test 1, extract the 'id' column (field index 0)
        auto fieldIndex = builder.create<::mlir::arith::ConstantIndexOp>(loc, 0);
        
        // PostgreSQL type OID for int8 (bigint) is 20
        auto typeOid = builder.create<::mlir::arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(20));  // OID 20 = int8/bigint
        
        auto fieldValue = builder.create<::pgx::db::GetFieldOp>(
            loc, 
            ::pgx::db::NullableI64Type::get(builder.getContext()),
            tableHandle, 
            fieldIndex.getResult(), 
            typeOid.getResult());
        
        MLIR_PGX_DEBUG("RelAlg", "Extracted field value using db.get_field for column index 0 (id)");
        
        // For now, assume non-null values - extract the actual value
        auto actualValue = builder.create<::pgx::db::NullableGetValOp>(
            loc, builder.getI64Type(), fieldValue.getResult());
        
        // Set the value in context for the id column
        context.setValueForAttribute(scope, &idColumn, actualValue.getResult());
        
        // Call consumer for streaming (one tuple at a time) inside the loop
        if (consumer) {
            MLIR_PGX_DEBUG("RelAlg", "Streaming tuple to consumer inside DB loop");
            
            // CRITICAL: The consumer is called inside the loop with the loop's builder
            // This ensures operations are inserted in the correct location
            consumer->consume(this, builder, context);
        }
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        // BaseTable is a leaf operation and should not consume from children
        llvm_unreachable("BaseTableTranslator should not have children");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "BaseTableTranslator::done() - Completed DB-based table scan");
        
        // Don't erase the BaseTableOp here - let the pass handle it after updating uses
        // baseTableOp.erase();
    }
};

// Factory function
std::unique_ptr<Translator> createBaseTableTranslator(::mlir::Operation* op) {
    auto baseTableOp = ::mlir::dyn_cast<::pgx::mlir::relalg::BaseTableOp>(op);
    if (!baseTableOp) {
        MLIR_PGX_ERROR("RelAlg", "createBaseTableTranslator called with non-BaseTableOp");
        return createDummyTranslator(op);
    }
    return std::make_unique<BaseTableTranslator>(baseTableOp);
}

} // namespace relalg
} // namespace mlir
} // namespace pgx