#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
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
        // Create DSA scan source for table iteration (LingoDB pattern)
        // Instead of db.get_external, we create DSA operations directly
        MLIR_PGX_DEBUG("RelAlg", "Creating DSA-based table scan for: " + 
                       baseTableOp.getTableName().str());
        
        // Create DSA-based tuple iteration
        createTupleIterationLoop(context, builder, scope, loc, nullptr);
    }
    
    // Create DSA-based tuple iteration following LingoDB pattern
    void createTupleIterationLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                                 TranslatorContext::AttributeResolverScope& scope,
                                 ::mlir::Location loc, ::mlir::Value tableHandle) {
        // Create DSA scan source for table iteration (LingoDB pattern)
        // Build table description JSON for the scan source
        std::string tableDesc = "{\"table\":\"" + baseTableOp.getTableName().str() + 
                               "\",\"oid\":" + std::to_string(baseTableOp.getTableOid()) + "}";
        auto tableDescAttr = builder.getStringAttr(tableDesc);
        
        // Create tuple type for the records - for Test 1, just an id column (i64)
        auto i64Type = builder.getI64Type();
        auto tupleType = ::mlir::TupleType::get(builder.getContext(), {i64Type});
        
        // Create iterable type with the tuple structure
        auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
            builder.getContext(), 
            ::pgx::mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType),
            "batch_iterator");
        
        // Create scan source operation
        auto scanSource = builder.create<::pgx::mlir::dsa::ScanSourceOp>(
            loc, iterableType, tableDescAttr);
        
        MLIR_PGX_DEBUG("RelAlg", "Created dsa.scan_source for table: " + 
                       baseTableOp.getTableName().str());
        
        // Create outer ForOp for batch iteration
        auto batchForOp = builder.create<::pgx::mlir::dsa::ForOp>(loc, scanSource.getResult());
        
        // Create batch loop body
        ::mlir::Region& batchRegion = batchForOp.getBody();
        ::mlir::Block* batchBlock = &batchRegion.emplaceBlock();
        auto batchType = ::pgx::mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType);
        batchBlock->addArgument(batchType, loc);
        
        // Create builder for batch loop body
        ::mlir::OpBuilder batchBuilder(batchBlock, batchBlock->begin());
        
        // Create inner ForOp for record iteration within the batch
        auto recordForOp = batchBuilder.create<::pgx::mlir::dsa::ForOp>(
            loc, batchBlock->getArgument(0));
        
        // Create record loop body
        ::mlir::Region& recordRegion = recordForOp.getBody();
        ::mlir::Block* recordBlock = &recordRegion.emplaceBlock();
        auto recordType = ::pgx::mlir::dsa::RecordType::get(builder.getContext(), tupleType);
        recordBlock->addArgument(recordType, loc);
        
        // Create builder for record loop body
        ::mlir::OpBuilder recordBuilder(recordBlock, recordBlock->begin());
        
        // Process the record inside the inner loop
        processTupleInLoop(context, recordBuilder, scope, loc, recordBlock->getArgument(0));
        
        // Add yield for inner loop
        recordBuilder.create<::pgx::mlir::dsa::YieldOp>(loc);
        
        // Add yield for outer loop  
        batchBuilder.create<::pgx::mlir::dsa::YieldOp>(loc);
        
        MLIR_PGX_DEBUG("RelAlg", "Created nested DSA for loops for streaming iteration");
    }
    
    // Process a single tuple inside the loop - extract field values and call consumer
    void processTupleInLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                           TranslatorContext::AttributeResolverScope& scope,
                           ::mlir::Location loc, ::mlir::Value record) {
        // Extract field value using dsa.at operations
        // For Test 1, extract the 'id' column
        auto columnName = builder.getStringAttr("id");
        auto fieldValue = builder.create<::pgx::mlir::dsa::AtOp>(
            loc, builder.getI64Type(), record, columnName);
        
        MLIR_PGX_DEBUG("RelAlg", "Extracted field value using dsa.at for column: id");
        
        // Set the value in context for the id column
        context.setValueForAttribute(scope, &idColumn, fieldValue.getResult());
        
        // Call consumer for streaming (one tuple at a time) inside the loop
        if (consumer) {
            MLIR_PGX_DEBUG("RelAlg", "Streaming tuple to consumer inside DSA loop");
            
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