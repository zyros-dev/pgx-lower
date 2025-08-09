#include <gtest/gtest.h>
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace pgx::mlir::relalg;

// Test that BaseTableTranslator initializes columns before produce()
TEST(ColumnInitializationTimingTest, BaseTableColumnsAvailableBeforeProduce) {
    // Setup MLIR context and dialects
    mlir::MLIRContext context;
    context.loadDialect<RelAlgDialect>();
    
    // Create a module and builder
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(module.getBody());
    
    // Create BaseTableOp
    mlir::Location loc = builder.getUnknownLoc();
    auto baseTableOp = builder.create<BaseTableOp>(
        loc,
        /*relationNameAttr=*/builder.getStringAttr("test"),
        /*tableOidAttr=*/builder.getI64IntegerAttr(12345)
    );
    
    // Create translator
    auto translator = Translator::createTranslator(baseTableOp.getOperation());
    ASSERT_NE(translator, nullptr);
    
    // Create TranslatorContext and initialize translator
    TranslatorContext translatorContext;
    translator->initializeWithContext(translatorContext);
    
    // CRITICAL TEST: getAvailableColumns() must work BEFORE produce() is called
    auto availableColumns = translator->getAvailableColumns();
    
    // Verify columns are available
    EXPECT_FALSE(availableColumns.empty()) << "BaseTableTranslator must have columns available before produce()";
    EXPECT_EQ(availableColumns.size(), 1) << "BaseTableTranslator should have exactly one column (id)";
    
    // Verify column details
    if (!availableColumns.empty()) {
        const Column* idColumn = *availableColumns.begin();
        EXPECT_NE(idColumn, nullptr);
        EXPECT_EQ(idColumn->getName(), "id");
        EXPECT_EQ(idColumn->getTableName(), "test");
        
        // Verify column type is i64
        auto columnType = idColumn->getType();
        EXPECT_TRUE(columnType.isa<mlir::IntegerType>());
        if (auto intType = columnType.dyn_cast<mlir::IntegerType>()) {
            EXPECT_EQ(intType.getWidth(), 64);
        }
    }
    
    // Cleanup
    module.erase();
}

// Test that column identity is preserved through ColumnManager
TEST(ColumnInitializationTimingTest, ColumnIdentityPreserved) {
    // Setup MLIR context and dialects
    mlir::MLIRContext context;
    context.loadDialect<RelAlgDialect>();
    
    // Create a module and builder
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(module.getBody());
    
    // Create BaseTableOp
    mlir::Location loc = builder.getUnknownLoc();
    auto baseTableOp = builder.create<BaseTableOp>(
        loc,
        /*relationNameAttr=*/builder.getStringAttr("test"),
        /*tableOidAttr=*/builder.getI64IntegerAttr(12345)
    );
    
    // Create two translators for the same table
    auto translator1 = Translator::createTranslator(baseTableOp.getOperation());
    auto translator2 = Translator::createTranslator(baseTableOp.getOperation());
    
    // Initialize both with the same context (shared ColumnManager)
    TranslatorContext translatorContext;
    translator1->initializeWithContext(translatorContext);
    translator2->initializeWithContext(translatorContext);
    
    // Get columns from both translators
    auto columns1 = translator1->getAvailableColumns();
    auto columns2 = translator2->getAvailableColumns();
    
    ASSERT_FALSE(columns1.empty());
    ASSERT_FALSE(columns2.empty());
    
    // Verify column identity is preserved (same pointer)
    const Column* idColumn1 = *columns1.begin();
    const Column* idColumn2 = *columns2.begin();
    
    // CRITICAL: Column pointers must be identical (shared via ColumnManager)
    EXPECT_EQ(idColumn1, idColumn2) << "Column identity must be preserved through ColumnManager";
    
    // Cleanup
    module.erase();
}

// Test that MaterializeTranslator can consume from BaseTableTranslator
TEST(ColumnInitializationTimingTest, ConsumerChainWithEarlyColumnInit) {
    // Setup MLIR context and dialects
    mlir::MLIRContext context;
    context.loadDialect<RelAlgDialect>();
    
    // Create a module and builder
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(module.getBody());
    
    // Create BaseTableOp
    mlir::Location loc = builder.getUnknownLoc();
    auto baseTableOp = builder.create<BaseTableOp>(
        loc,
        /*relationNameAttr=*/builder.getStringAttr("test"),
        /*tableOidAttr=*/builder.getI64IntegerAttr(12345)
    );
    
    // Create MaterializeOp that consumes from BaseTableOp
    auto materializeOp = builder.create<MaterializeOp>(
        loc,
        /*rel=*/baseTableOp.getResult()
    );
    
    // Create translators
    auto baseTranslator = Translator::createTranslator(baseTableOp.getOperation());
    auto materializeTranslator = Translator::createTranslator(materializeOp.getOperation());
    
    // Initialize with context
    TranslatorContext translatorContext;
    baseTranslator->initializeWithContext(translatorContext);
    materializeTranslator->initializeWithContext(translatorContext);
    
    // Set up consumer chain
    baseTranslator->setInfo(materializeTranslator.get(), ColumnSet());
    
    // Verify BaseTable has columns available
    auto baseColumns = baseTranslator->getAvailableColumns();
    EXPECT_FALSE(baseColumns.empty()) << "BaseTable must have columns after initialization";
    
    // Verify MaterializeTranslator can see required columns
    // This tests that the consumer chain works with early column initialization
    materializeTranslator->setInfo(nullptr, baseColumns);
    
    // The test passes if no assertions fail
    // In a real scenario, produce() would be called and the consumer chain would execute
    
    // Cleanup
    module.erase();
}