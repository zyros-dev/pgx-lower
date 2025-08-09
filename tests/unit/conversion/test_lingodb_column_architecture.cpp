// Test for LingoDB-style column management architecture
// Verifies column identity sharing between translators

#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

using namespace pgx::mlir::relalg;

class LingoDBColumnArchitectureTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<::mlir::arith::ArithDialect>();
        builder = std::make_unique<::mlir::OpBuilder>(&context);
    }

    ::mlir::MLIRContext context;
    std::unique_ptr<::mlir::OpBuilder> builder;
};

TEST_F(LingoDBColumnArchitectureTest, ColumnIdentitySharing) {
    PGX_INFO("Testing column identity sharing in ColumnManager");
    
    // Create ColumnManager and TranslatorContext
    auto columnManager = std::make_shared<ColumnManager>();
    columnManager->setContext(&context);
    
    TranslatorContext translatorContext;
    // Note: TranslatorContext creates its own ColumnManager in constructor
    // We need to use that one for proper integration
    auto ctxColumnManager = translatorContext.getColumnManager();
    ctxColumnManager->setContext(&context);
    
    // Get same column from multiple requests
    auto i64Type = ::mlir::IntegerType::get(&context, 64);
    auto baseColumn = ctxColumnManager->get("test", "id", i64Type);
    auto materializeColumn = ctxColumnManager->get("test", "id", i64Type);
    
    // Verify pointer identity (not just equality)
    EXPECT_EQ(baseColumn.get(), materializeColumn.get()) 
        << "Same column should return identical pointers";
    EXPECT_EQ(baseColumn, materializeColumn) 
        << "shared_ptr instances should be equal";
    
    // Verify column has correct type
    EXPECT_EQ(baseColumn->type, i64Type) 
        << "Column should have i64 type";
    
    // Log for debugging
    PGX_INFO("Column pointer: " + std::to_string(reinterpret_cast<uintptr_t>(baseColumn.get())));
}

TEST_F(LingoDBColumnArchitectureTest, ScopedSymbolResolution) {
    PGX_INFO("Testing scoped symbol resolution in TranslatorContext");
    
    TranslatorContext context;
    auto columnManager = context.getColumnManager();
    columnManager->setContext(&this->context);
    
    // Create columns
    auto i64Type = ::mlir::IntegerType::get(&this->context, 64);
    auto column1 = columnManager->get("test", "id", i64Type);
    auto column2 = columnManager->get("test", "name", i64Type);
    
    // Create dummy MLIR values
    auto loc = builder->getUnknownLoc();
    auto value1 = builder->create<::mlir::arith::ConstantOp>(loc, builder->getI64IntegerAttr(1));
    auto value2 = builder->create<::mlir::arith::ConstantOp>(loc, builder->getI64IntegerAttr(2));
    
    // Test scoped resolution
    {
        auto scope = context.createScope();
        context.setValueForAttribute(scope, column1.get(), value1.getResult());
        context.setValueForAttribute(scope, column2.get(), value2.getResult());
        
        // Verify resolution works
        auto resolved1 = context.getValueForAttribute(column1.get());
        auto resolved2 = context.getValueForAttribute(column2.get());
        
        EXPECT_EQ(resolved1, value1.getResult()) << "Column1 should resolve to value1";
        EXPECT_EQ(resolved2, value2.getResult()) << "Column2 should resolve to value2";
    }
    
    // After scope exit, columns should not be resolvable
    EXPECT_EQ(context.getUnsafeValueForAttribute(column1.get()), nullptr)
        << "Column1 should not be resolvable after scope exit";
}

TEST_F(LingoDBColumnArchitectureTest, ColumnManagerReverseLookup) {
    PGX_INFO("Testing ColumnManager reverse lookup");
    
    auto columnManager = std::make_shared<ColumnManager>();
    columnManager->setContext(&context);
    
    // Create column
    auto i64Type = ::mlir::IntegerType::get(&context, 64);
    auto column = columnManager->get("orders", "order_id", i64Type);
    
    // Test reverse lookup
    auto [scope, name] = columnManager->getName(column.get());
    
    EXPECT_EQ(scope, "orders") << "Scope should be 'orders'";
    EXPECT_EQ(name, "order_id") << "Name should be 'order_id'";
}

TEST_F(LingoDBColumnArchitectureTest, UniqueScopeGeneration) {
    PGX_INFO("Testing unique scope generation");
    
    auto columnManager = std::make_shared<ColumnManager>();
    
    // Test scope uniquification
    auto scope1 = columnManager->getUniqueScope("subquery");
    auto scope2 = columnManager->getUniqueScope("subquery");
    auto scope3 = columnManager->getUniqueScope("subquery");
    
    EXPECT_EQ(scope1, "subquery") << "First scope should be base name";
    EXPECT_EQ(scope2, "subquery1") << "Second scope should have suffix";
    EXPECT_EQ(scope3, "subquery2") << "Third scope should have incremented suffix";
    
    // Different base should start fresh
    auto other1 = columnManager->getUniqueScope("join");
    EXPECT_EQ(other1, "join") << "New base should start without suffix";
}