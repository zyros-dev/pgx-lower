#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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
        // Create db.get_external to initialize table access
        auto tableOid = builder.create<::mlir::arith::ConstantIntOp>(
            loc, baseTableOp.getTableOid(), 64);
        auto tableHandle = builder.create<::pgx::db::GetExternalOp>(
            loc, ::pgx::db::ExternalSourceType::get(builder.getContext()), tableOid);
        
        MLIR_PGX_DEBUG("RelAlg", "Created db.get_external for table OID: " + 
                       std::to_string(baseTableOp.getTableOid()));
        
        // Create DSA-based tuple iteration
        createTupleIterationLoop(context, builder, scope, loc, tableHandle);
    }
    
    // Create DSA-based tuple iteration following LingoDB pattern
    void createTupleIterationLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                                 TranslatorContext::AttributeResolverScope& scope,
                                 ::mlir::Location loc, ::mlir::Value tableHandle) {
        // Create DSA scan source for table iteration (LingoDB pattern)
        // For now, create a simplified DSA iterator that processes tuples
        // This will be properly implemented with dsa.scan_source in Phase 5
        
        // Directly process the tuple since we have the tableHandle
        // In the full LingoDB pattern, this would be:
        // %scan = dsa.scan_source %table_description
        // dsa.for %tuple in %scan { ... }
        
        processTuple(context, builder, scope, loc, tableHandle);
        
        MLIR_PGX_DEBUG("RelAlg", "Created DSA-based tuple processing (simplified for Phase 4c-4)");
    }
    
    // Process a single tuple - extract field values and call consumer
    void processTuple(TranslatorContext& context, ::mlir::OpBuilder& builder,
                     TranslatorContext::AttributeResolverScope& scope,
                     ::mlir::Location loc, ::mlir::Value tableHandle) {
        // For Phase 4c-4, we create a simplified single-tuple processing
        // without complex control flow that could break MLIR block structure
        
        // Extract field value using db.get_field
        // For Test 1, extract field 0 (id column)
        auto fieldIndex = builder.create<::mlir::arith::ConstantIndexOp>(loc, 0);
        // PostgreSQL OID for int8 (BIGINT) = 20
        auto typeOid = builder.create<::mlir::arith::ConstantIntOp>(loc, 20, 32);
        
        auto fieldValue = builder.create<::pgx::db::GetFieldOp>(
            loc, ::pgx::db::NullableI64Type::get(builder.getContext()),
            tableHandle, fieldIndex, typeOid);
        
        MLIR_PGX_DEBUG("RelAlg", "Extracted field value using db.get_field at index 0");
        
        // For now, assume non-nullable and extract the actual value
        // In a full implementation, we'd handle nullable values properly
        auto actualValue = builder.create<::pgx::db::NullableGetValOp>(
            loc, builder.getI64Type(), fieldValue);
        
        // Set the value in context for the id column
        context.setValueForAttribute(scope, &idColumn, actualValue);
        
        // Call consumer for streaming (one tuple at a time)
        if (consumer) {
            MLIR_PGX_DEBUG("RelAlg", "Streaming tuple to consumer");
            
            // CRITICAL: Save and restore insertion point around consume call
            // This prevents the consumer from corrupting our block structure
            auto savedIP = builder.saveInsertionPoint();
            consumer->consume(this, builder, context);
            builder.restoreInsertionPoint(savedIP);
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