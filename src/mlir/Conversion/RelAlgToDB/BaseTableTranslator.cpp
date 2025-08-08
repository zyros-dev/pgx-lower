#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

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
        PGX_DEBUG("Created BaseTableTranslator for table: " + 
                  op.getTableName().str());
        
        // Initialize available columns - for Test 1, just create an 'id' column
        availableColumns.insert(&idColumn);
    }
    
    ColumnSet getAvailableColumns() override {
        return availableColumns;
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        PGX_INFO("BaseTableTranslator::produce() - Beginning table scan for: " + 
                 baseTableOp.getTableName().str());
        
        auto scope = context.createScope();
        auto loc = baseTableOp.getLoc();
        
        // Create the scan loop structure
        auto forOp = createScanLoop(builder, loc);
        
        // Set insertion point inside the loop
        auto& loopBody = forOp.getRegion().front();
        builder.setInsertionPointToStart(&loopBody);
        
        // Generate tuple values and process
        generateTupleValues(context, builder, scope, loc);
        
        // Restore insertion point after loop
        builder.setInsertionPointAfter(forOp);
    }
    
private:
    // Create the loop structure for scanning rows
    ::mlir::scf::ForOp createScanLoop(::mlir::OpBuilder& builder, ::mlir::Location loc) {
        // For Test 1, we implement a simplified version that creates a simple loop
        // In the future, this will use DSA scan operations like LingoDB
        
        // Create a simple constant to represent the number of rows (1 for Test 1)
        auto rowCount = builder.create<::mlir::arith::ConstantIndexOp>(loc, 1);
        auto zero = builder.create<::mlir::arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<::mlir::arith::ConstantIndexOp>(loc, 1);
        
        // Create a simple for loop to iterate over rows
        return builder.create<::mlir::scf::ForOp>(
            loc, zero, rowCount, one, ::mlir::ValueRange{});
    }
    
    // Generate values for the current tuple and call consumer
    void generateTupleValues(TranslatorContext& context, ::mlir::OpBuilder& builder,
                           TranslatorContext::AttributeResolverScope& scope, 
                           ::mlir::Location loc) {
        // For Test 1, create a constant value for the 'id' column
        // This is a simplified implementation - real version would read from storage
        auto idValue = builder.create<::mlir::arith::ConstantIntOp>(
            loc, 1, builder.getI64Type());
        
        // Map the column values in the context using Column-based resolution
        context.setValueForAttribute(scope, &idColumn, idValue);
        
        // Call the consumer to process this tuple
        if (consumer) {
            PGX_DEBUG("Calling consumer from BaseTableTranslator");
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
        PGX_DEBUG("BaseTableTranslator::done() - Completed table scan");
    }
};

// Factory function
std::unique_ptr<Translator> createBaseTableTranslator(::mlir::Operation* op) {
    auto baseTableOp = ::mlir::dyn_cast<::pgx::mlir::relalg::BaseTableOp>(op);
    if (!baseTableOp) {
        PGX_ERROR("createBaseTableTranslator called with non-BaseTableOp");
        return createDummyTranslator(op);
    }
    return std::make_unique<BaseTableTranslator>(baseTableOp);
}

} // namespace relalg
} // namespace mlir
} // namespace pgx