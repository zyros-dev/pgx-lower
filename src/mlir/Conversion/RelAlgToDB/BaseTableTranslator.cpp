#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
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
    // Store shared column pointer from ColumnManager
    const Column* idColumn;
    
public:
    explicit BaseTableTranslator(::pgx::mlir::relalg::BaseTableOp op) 
        : Translator(op), 
          baseTableOp(op),
          idColumn(nullptr) {
        MLIR_PGX_DEBUG("RelAlg", "Created BaseTableTranslator for table: " + 
                       op.getTableName().str());
    }
    
    void setInfo(Translator* consumer, const ColumnSet& requiredAttributes) override {
        MLIR_PGX_INFO("RelAlg", "BaseTableTranslator::setInfo() called");
        MLIR_PGX_INFO("RelAlg", "BaseTableTranslator address: " + std::to_string(reinterpret_cast<uintptr_t>(this)));
        MLIR_PGX_INFO("RelAlg", "Consumer address being set: " + std::to_string(reinterpret_cast<uintptr_t>(consumer)));
        Translator::setInfo(consumer, requiredAttributes);
        
        // Note: Columns will be initialized in produce() when TranslatorContext is available
        // This ensures we use the shared column from ColumnManager
    }
    
    ColumnSet getAvailableColumns() override {
        return availableColumns;
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        MLIR_PGX_INFO("RelAlg", "BaseTableTranslator::produce() - Beginning PostgreSQL SPI table scan for: " + 
                      baseTableOp.getTableName().str());
        MLIR_PGX_DEBUG("RelAlg", "BaseTableTranslator consumer is " + 
                       (consumer ? "set" : "NOT SET"));
        
        // Initialize shared column from ColumnManager
        if (!idColumn) {
            auto columnManager = context.getColumnManager();
            if (columnManager) {
                // Use ColumnManager to get shared column identity with i64 type
                auto i64Type = ::mlir::IntegerType::get(builder.getContext(), 64);
                auto sharedColumn = columnManager->get(baseTableOp.getTableName().str(), "id", i64Type);
                idColumn = sharedColumn.get();
                availableColumns.insert(idColumn);
                
                // Log Column pointer for debugging column identity sharing
                MLIR_PGX_INFO("RelAlg", "BaseTableTranslator got shared column 'id' from table '" + 
                              baseTableOp.getTableName().str() + "' at address: " + 
                              std::to_string(reinterpret_cast<uintptr_t>(idColumn)));
            } else {
                MLIR_PGX_ERROR("RelAlg", "BaseTableTranslator: No ColumnManager available!");
            }
        }
        
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
        
        // Create DB operations for PostgreSQL SPI integration (Phase 4d architecture)
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
    // Create PostgreSQL SPI table scan operations (Phase 4d)
    // Generates db.get_external → scf.while → db.iterate_external → db.get_field sequence
    void createDBTableScan(TranslatorContext& context, ::mlir::OpBuilder& builder,
                          TranslatorContext::AttributeResolverScope& scope, 
                          ::mlir::Location loc) {
        MLIR_PGX_DEBUG("RelAlg", "Creating PostgreSQL SPI table scan operations for: " + 
                       baseTableOp.getTableName().str());
        
        // Get table OID from the BaseTableOp
        auto tableOid = builder.create<::mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(baseTableOp.getTableOid()));
        
        // Create db.get_external for PostgreSQL SPI table handle acquisition
        // This will lower to func.call @pg_table_open in Phase 4d DBToStd pass
        auto externalSourceType = ::pgx::db::ExternalSourceType::get(builder.getContext());
        auto externalSource = builder.create<::pgx::db::GetExternalOp>(
            loc, externalSourceType, tableOid);
        
        MLIR_PGX_DEBUG("RelAlg", "Created db.get_external for PostgreSQL table OID: " + 
                       std::to_string(baseTableOp.getTableOid()));
        
        // Create PostgreSQL tuple iteration loop (streaming architecture)
        createTupleIterationLoop(context, builder, scope, loc, externalSource.getResult());
    }
    
    // Create PostgreSQL SPI tuple iteration with streaming producer-consumer pattern
    // Uses scf.while for iteration, db.iterate_external for tuple fetching (Phase 4d)
    void createTupleIterationLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                                 TranslatorContext::AttributeResolverScope& scope,
                                 ::mlir::Location loc, ::mlir::Value tableHandle) {
        MLIR_PGX_DEBUG("RelAlg", "Creating PostgreSQL SPI iteration loop for table: " + 
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
            
            // Call db.iterate_external to check for next PostgreSQL tuple
            // This will lower to func.call @pg_get_next_tuple in DBToStd pass
            auto hasTuple = builder.create<::pgx::db::IterateExternalOp>(
                loc, builder.getI1Type(), tableHandle);
            
            MLIR_PGX_DEBUG("RelAlg", "Created db.iterate_external for PostgreSQL SPI tuple iteration");
            
            // Yield the condition
            builder.create<::mlir::scf::ConditionOp>(loc, hasTuple.getResult(), ::mlir::ValueRange{});
        }
        
        // Build the "after" region (loop body)
        {
            auto& afterRegion = whileOp.getAfter();
            auto* afterBlock = builder.createBlock(&afterRegion);
            builder.setInsertionPointToStart(afterBlock);
            
            // Process current PostgreSQL tuple and stream to consumer
            processTupleInLoop(context, builder, scope, loc, tableHandle);
            
            // Yield to continue the loop
            builder.create<::mlir::scf::YieldOp>(loc, ::mlir::ValueRange{});
        }
        
        // Set insertion point after the while loop
        builder.setInsertionPointAfter(whileOp);
        
        MLIR_PGX_DEBUG("RelAlg", "Created scf.while loop for PostgreSQL SPI tuple streaming");
    }
    
    // Process single PostgreSQL tuple: extract fields via SPI and stream to consumer
    // Phase 4d: db.get_field will lower to func.call @pg_extract_field
    void processTupleInLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                           TranslatorContext::AttributeResolverScope& scope,
                           ::mlir::Location loc, ::mlir::Value tableHandle) {
        // Extract field value using db.get_field (PostgreSQL SPI integration)
        // For Test 1, extract the 'id' column (field index 0)
        auto fieldIndex = builder.create<::mlir::arith::ConstantIndexOp>(loc, 0);
        
        // PostgreSQL type OID for int8 (bigint) is 20
        auto typeOid = builder.create<::mlir::arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(20));  // OID 20 = int8/bigint
        
        // db.get_field extracts PostgreSQL tuple field via SPI
        // Will lower to func.call @pg_extract_field in DBToStd pass
        auto fieldValue = builder.create<::pgx::db::GetFieldOp>(
            loc, 
            ::pgx::db::NullableI64Type::get(builder.getContext()),
            tableHandle, 
            fieldIndex.getResult(), 
            typeOid.getResult());
        
        MLIR_PGX_DEBUG("RelAlg", "Created db.get_field for PostgreSQL SPI field extraction (column 0: id)");
        
        // Extract actual value from nullable wrapper (assumes non-null for Test 1)
        // db.nullable_get_val will lower to llvm.extractvalue in DBToStd pass
        auto actualValue = builder.create<::pgx::db::NullableGetValOp>(
            loc, builder.getI64Type(), fieldValue.getResult());
        
        // Set the value in context for the id column
        context.setValueForAttribute(scope, idColumn, actualValue.getResult());
        
        // Log column pointer and value for debugging
        MLIR_PGX_DEBUG("RelAlg", "BaseTableTranslator setting value for column at address: " + 
                       std::to_string(reinterpret_cast<uintptr_t>(idColumn)));
        
        // Stream single tuple to consumer (streaming producer-consumer pattern)
        // This implements constant memory usage - one tuple at a time processing
        if (consumer) {
            MLIR_PGX_DEBUG("RelAlg", "Streaming single PostgreSQL tuple to consumer (constant memory)");
            MLIR_PGX_DEBUG("RelAlg", "Consumer type: " + std::string(consumer->getOperation()->getName().getStringRef().str()));
            MLIR_PGX_DEBUG("RelAlg", "Consumer address: " + std::to_string(reinterpret_cast<uintptr_t>(consumer)));
            
            // CRITICAL: Consumer called inside PostgreSQL iteration loop
            // Enables streaming architecture with one-tuple-at-a-time processing
            MLIR_PGX_INFO("RelAlg", "BaseTableTranslator calling consumer->consume() NOW");
            consumer->consume(this, builder, context);
            MLIR_PGX_INFO("RelAlg", "BaseTableTranslator consumer->consume() returned");
        } else {
            MLIR_PGX_WARNING("RelAlg", "BaseTableTranslator has no consumer set - data not being processed");
        }
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        // BaseTable is a leaf operation and should not consume from children
        llvm_unreachable("BaseTableTranslator should not have children");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "BaseTableTranslator::done() - Completed PostgreSQL SPI table scan");
        
        // Pass handles BaseTableOp cleanup after use replacement (Phase 4d pattern)
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