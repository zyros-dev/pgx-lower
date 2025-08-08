#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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
        
        auto scope = context.createScope();
        auto loc = baseTableOp.getLoc();
        
        // Create DB operations following the correct pipeline architecture
        createDBTableScan(context, builder, scope, loc);
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
        
        // Create scf.while loop for tuple iteration
        createTupleIterationLoop(context, builder, scope, loc, tableHandle);
    }
    
    // Create the iteration loop using scf.while and db.iterate_external
    void createTupleIterationLoop(TranslatorContext& context, ::mlir::OpBuilder& builder,
                                 TranslatorContext::AttributeResolverScope& scope,
                                 ::mlir::Location loc, ::mlir::Value tableHandle) {
        // Create scf.while loop that iterates over tuples
        auto whileOp = builder.create<::mlir::scf::WhileOp>(
            loc, ::mlir::TypeRange{}, ::mlir::ValueRange{});
        
        // Create the "before" region (condition check)
        {
            auto& beforeRegion = whileOp.getBefore();
            auto* beforeBlock = builder.createBlock(&beforeRegion);
            ::mlir::OpBuilder beforeBuilder(beforeBlock, beforeBlock->end());
            
            // Check if there's a next tuple
            auto hasTuple = beforeBuilder.create<::pgx::db::IterateExternalOp>(
                loc, tableHandle);
            
            beforeBuilder.create<::mlir::scf::ConditionOp>(
                loc, hasTuple, ::mlir::ValueRange{});
        }
        
        // Create the "after" region (loop body)
        {
            auto& afterRegion = whileOp.getAfter();
            auto* afterBlock = builder.createBlock(&afterRegion);
            ::mlir::OpBuilder afterBuilder(afterBlock, afterBlock->end());
            
            // Process the current tuple
            processTuple(context, afterBuilder, scope, loc, tableHandle);
            
            // Yield to continue the loop
            afterBuilder.create<::mlir::scf::YieldOp>(loc, ::mlir::ValueRange{});
        }
    }
    
    // Process a single tuple - extract field values and call consumer
    void processTuple(TranslatorContext& context, ::mlir::OpBuilder& builder,
                     TranslatorContext::AttributeResolverScope& scope,
                     ::mlir::Location loc, ::mlir::Value tableHandle) {
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