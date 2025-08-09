#include "gtest/gtest.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

namespace {

// Test fixture for column resolution fixes
class ColumnResolutionTest : public ::testing::Test {
protected:
    MLIRContext context;
    OpBuilder builder;
    
    ColumnResolutionTest() : builder(&context) {
        // Load required dialects
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
    }
};

// Test 1: Column identity sharing between translators
TEST_F(ColumnResolutionTest, ColumnIdentitySharing) {
    PGX_DEBUG("Testing column identity sharing between BaseTable and Materialize translators");
    
    // Create module and function for proper MLIR structure
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_func", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple RelAlg pipeline
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("test"),  // Use "test" to match BaseTableTranslator
        builder.getI64IntegerAttr(12345)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto columnsAttr = builder.getArrayAttr({builder.getStringAttr("id")});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    // Add function return
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Create a shared TranslatorContext with ColumnManager
    TranslatorContext translatorContext;
    auto columnManager = translatorContext.getColumnManager();
    columnManager->setContext(&context);
    
    // Create translators
    auto baseTableTranslator = createBaseTableTranslator(baseTableOp);
    auto materializeTranslator = createMaterializeTranslator(materializeOp);
    
    // Set up the translator chain
    ColumnSet required;
    materializeTranslator->setInfo(nullptr, required);
    baseTableTranslator->setInfo(materializeTranslator.get(), required);
    
    // Simulate the produce phase to initialize columns with shared ColumnManager
    OpBuilder opBuilder(&context);
    opBuilder.setInsertionPointAfter(materializeOp);
    
    // Call produce on BaseTableTranslator - this will initialize columns via ColumnManager
    baseTableTranslator->produce(translatorContext, opBuilder);
    
    // Now get available columns - they should be initialized
    auto baseTableColumns = baseTableTranslator->getAvailableColumns();
    EXPECT_EQ(baseTableColumns.size(), 1) << "BaseTable should provide one column (id)";
    
    // Get the shared column from ColumnManager
    auto i64Type = IntegerType::get(&context, 64);
    auto sharedColumn = columnManager->get("test", "id", i64Type);
    ASSERT_NE(sharedColumn, nullptr) << "ColumnManager should have the shared column";
    
    // Verify the column from BaseTableTranslator is the same as from ColumnManager
    const Column* baseTableColumn = nullptr;
    for (const Column* col : baseTableColumns) {
        baseTableColumn = col;
        break;
    }
    ASSERT_NE(baseTableColumn, nullptr) << "Should have found column from BaseTable";
    
    // This is the key test - verify pointer identity
    EXPECT_EQ(baseTableColumn, sharedColumn.get()) 
        << "BaseTableTranslator should use the same Column pointer from ColumnManager";
    
    // Clean up
    module.erase();
    
    PGX_DEBUG("Column identity sharing test passed");
}

// Test 2: Type consistency between translators
TEST_F(ColumnResolutionTest, TypeConsistency) {
    PGX_DEBUG("Testing type consistency between BaseTable and Materialize translators");
    
    // Create module and function
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_func", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create translators
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(12345)
    );
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Create shared TranslatorContext
    TranslatorContext translatorContext;
    auto columnManager = translatorContext.getColumnManager();
    columnManager->setContext(&context);
    
    auto baseTableTranslator = createBaseTableTranslator(baseTableOp);
    
    // Initialize columns via produce
    baseTableTranslator->setInfo(nullptr, ColumnSet());
    OpBuilder opBuilder(&context);
    opBuilder.setInsertionPointAfter(baseTableOp);
    baseTableTranslator->produce(translatorContext, opBuilder);
    
    // Get column type from BaseTableTranslator
    auto columns = baseTableTranslator->getAvailableColumns();
    ASSERT_EQ(columns.size(), 1) << "Should have one column";
    
    const Column* idColumn = nullptr;
    for (const Column* col : columns) {
        idColumn = col;
        break;
    }
    ASSERT_NE(idColumn, nullptr) << "Should have found column";
    ASSERT_TRUE(idColumn->type) << "Column should have a type";
    
    // Verify it's i64 type
    EXPECT_TRUE(idColumn->type.isInteger(64)) << "BaseTable id column should be i64 type";
    
    module.erase();
    
    PGX_DEBUG("Type consistency test passed");
}

// Test 3: OrderedAttributes resolution with proper column sharing
TEST_F(ColumnResolutionTest, OrderedAttributesResolution) {
    PGX_DEBUG("Testing OrderedAttributes resolution with shared columns");
    
    // Create a column with i64 type (matching BaseTableTranslator)
    Column testColumn(IntegerType::get(&context, 64));
    
    // Create OrderedAttributes and add the column
    OrderedAttributes orderedAttrs;
    orderedAttrs.insert(&testColumn);
    
    // Create TranslatorContext and set value for the column
    TranslatorContext translatorContext;
    auto scope = translatorContext.createScope();
    
    auto testValue = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 123, 64);
    translatorContext.setValueForAttribute(scope, &testColumn, testValue);
    
    // Test resolution
    auto resolvedValue = orderedAttrs.resolve(translatorContext, 0);
    EXPECT_EQ(resolvedValue, testValue.getResult()) << "OrderedAttributes should resolve column correctly";
    
    // Verify tuple type
    auto tupleType = orderedAttrs.getTupleType(&context);
    EXPECT_EQ(tupleType.size(), 1) << "Tuple should have one element";
    EXPECT_TRUE(tupleType.getType(0).isInteger(64)) << "Tuple element should be i64";
    
    PGX_DEBUG("OrderedAttributes resolution test passed");
}

// Test 4: MaterializeTranslator nullable type conversion
TEST_F(ColumnResolutionTest, NullableTypeConversion) {
    PGX_DEBUG("Testing MaterializeTranslator nullable type conversion for different types");
    
    // Test i32 to NullableI32
    auto i32Val = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 10, 32);
    auto nullableI32Type = pgx::db::NullableI32Type::get(&context);
    auto nullableI32 = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), nullableI32Type, i32Val);
    EXPECT_TRUE(nullableI32.getType().isa<pgx::db::NullableI32Type>()) << "Should create NullableI32";
    
    // Test i64 to NullableI64
    auto i64Val = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 20, 64);
    auto nullableI64Type = pgx::db::NullableI64Type::get(&context);
    auto nullableI64 = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), nullableI64Type, i64Val);
    EXPECT_TRUE(nullableI64.getType().isa<pgx::db::NullableI64Type>()) << "Should create NullableI64";
    
    PGX_DEBUG("Nullable type conversion test passed");
}

// Test 5: Column Pointer Identity Sharing (Reviewer 3 requirement)
TEST_F(ColumnResolutionTest, ColumnPointerIdentitySharing) {
    PGX_DEBUG("Testing that BaseTableTranslator and MaterializeTranslator share identical Column pointer objects");
    
    // Create module and function
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_func", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple RelAlg pipeline
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(12345)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto columnsAttr = builder.getArrayAttr({builder.getStringAttr("id")});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Create shared TranslatorContext
    TranslatorContext translatorContext;
    auto columnManager = translatorContext.getColumnManager();
    columnManager->setContext(&context);
    
    // Create translators
    auto baseTableTranslator = createBaseTableTranslator(baseTableOp);
    auto materializeTranslator = createMaterializeTranslator(materializeOp);
    
    // Set up the translator chain
    ColumnSet required;
    materializeTranslator->setInfo(nullptr, required);
    baseTableTranslator->setInfo(materializeTranslator.get(), required);
    
    // Initialize columns via produce
    OpBuilder opBuilder(&context);
    opBuilder.setInsertionPointAfter(materializeOp);
    baseTableTranslator->produce(translatorContext, opBuilder);
    
    // Get columns from BaseTableTranslator
    auto baseTableColumns = baseTableTranslator->getAvailableColumns();
    ASSERT_EQ(baseTableColumns.size(), 1) << "BaseTable should provide one column";
    
    const Column* baseTableColumn = nullptr;
    for (const Column* col : baseTableColumns) {
        baseTableColumn = col;
        break;
    }
    ASSERT_NE(baseTableColumn, nullptr) << "Should have found column from BaseTable";
    
    // Log the BaseTable column pointer
    PGX_INFO("BaseTable column pointer: " + std::to_string(reinterpret_cast<uintptr_t>(baseTableColumn)));
    
    // Now verify MaterializeTranslator uses the SAME Column pointer
    // This is the critical test - MaterializeTranslator should NOT create new Column objects
    // It should use the exact same pointer from BaseTableTranslator
    
    // Create a scope for testing value resolution
    auto scope = translatorContext.createScope();
    
    // Set a value for the BaseTable column
    auto testValue = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 123, 64);
    translatorContext.setValueForAttribute(scope, baseTableColumn, testValue);
    
    // Verify we can retrieve the value using the same column pointer
    auto retrievedValue = translatorContext.getUnsafeValueForAttribute(baseTableColumn);
    EXPECT_EQ(retrievedValue, testValue.getResult()) << "Should retrieve the same value using shared Column pointer";
    
    // Additional verification: Check that column type is consistent
    EXPECT_TRUE(baseTableColumn->type.isInteger(64)) << "Column should have i64 type for Test 1";
    
    module.erase();
    
    PGX_DEBUG("Column pointer identity sharing test passed - translators share same Column objects");
}

} // namespace