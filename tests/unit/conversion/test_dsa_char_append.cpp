#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include "pgx-lower/execution/logging.h"

class DSACharAppendTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    DSACharAppendTest() : builder(&context) {
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(module.getBody());
    }

    ~DSACharAppendTest() {
        if (module) {
            module.erase();
        }
    }
};

TEST_F(DSACharAppendTest, Handle40BitInteger) {
    // Test that 40-bit integers (from db.char<5>) are handled correctly
    auto loc = builder.getUnknownLoc();
    
    // Create a function to test in
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_func", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a 40-bit integer type (simulating db.char<5>)
    auto i40Type = builder.getIntegerType(40);
    
    // Create a constant 40-bit value
    auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 0x48656C6C6F, i40Type); // "Hello" in hex
    
    // Create a table builder - needs a row type
    auto rowType = mlir::TupleType::get(&context, {i40Type});
    auto tableBuilderType = mlir::dsa::TableBuilderType::get(&context, rowType);
    auto builderOp = builder.create<mlir::dsa::CreateDS>(loc, tableBuilderType);
    
    // Create a valid flag (true)
    auto validValue = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    
    // Test the append operation with 40-bit integer
    // Append doesn't return a result - it modifies the builder in place
    auto appendOp = builder.create<mlir::dsa::Append>(
        loc,
        builderOp.getResult(),
        value.getResult(),
        validValue.getResult()
    );
    
    // Verify the operation was created successfully
    ASSERT_NE(appendOp, nullptr);
    
    // Verify the operand types
    EXPECT_EQ(appendOp.getVal().getType(), i40Type);
    EXPECT_TRUE(appendOp.getDs().getType().isa<mlir::dsa::TableBuilderType>());
    
    // Add return
    builder.create<mlir::func::ReturnOp>(loc);
    
    PGX_INFO("DSACharAppendTest: Successfully created append operation with 40-bit integer");
}

TEST_F(DSACharAppendTest, HandleVariousNonStandardWidths) {
    // Test various non-standard integer widths
    auto loc = builder.getUnknownLoc();
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_various_widths", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Test widths that should use the default case
    std::vector<unsigned> testWidths = {24, 40, 48, 56, 72, 96};
    
    // Create row type with mixed integer types
    std::vector<mlir::Type> columnTypes;
    for (unsigned width : testWidths) {
        columnTypes.push_back(builder.getIntegerType(width));
    }
    auto rowType = mlir::TupleType::get(&context, columnTypes);
    auto tableBuilderType = mlir::dsa::TableBuilderType::get(&context, rowType);
    auto builderOp = builder.create<mlir::dsa::CreateDS>(loc, tableBuilderType);
    auto validValue = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    
    mlir::Value currentBuilder = builderOp.getResult();
    
    for (unsigned width : testWidths) {
        // Create integer type with non-standard width
        auto intType = builder.getIntegerType(width);
        
        // Create a constant value
        auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 42, intType);
        
        // Create append operation (no result type - modifies builder in place)
        auto appendOp = builder.create<mlir::dsa::Append>(
            loc,
            currentBuilder,
            value.getResult(),
            validValue.getResult()
        );
        
        ASSERT_NE(appendOp, nullptr) << "Failed to create append for width " << width;
        
        // Continue using the same builder for next operation
        
        PGX_DEBUG(std::string("DSACharAppendTest: Created append for ") + std::to_string(width) + "-bit integer");
    }
    
    builder.create<mlir::func::ReturnOp>(loc);
    
    PGX_INFO("DSACharAppendTest: Successfully tested various non-standard integer widths");
}

TEST_F(DSACharAppendTest, HandleStandardWidths) {
    // Verify standard widths still work correctly
    auto loc = builder.getUnknownLoc();
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_standard_widths", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Test standard widths that have specific cases
    std::vector<unsigned> standardWidths = {8, 16, 32, 64, 128};
    
    // Create row type with standard integer types
    std::vector<mlir::Type> columnTypes;
    for (unsigned width : standardWidths) {
        columnTypes.push_back(builder.getIntegerType(width));
    }
    auto rowType = mlir::TupleType::get(&context, columnTypes);
    auto tableBuilderType = mlir::dsa::TableBuilderType::get(&context, rowType);
    auto builderOp = builder.create<mlir::dsa::CreateDS>(loc, tableBuilderType);
    auto validValue = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    
    mlir::Value currentBuilder = builderOp.getResult();
    
    for (unsigned width : standardWidths) {
        auto intType = builder.getIntegerType(width);
        auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 100, intType);
        
        auto appendOp = builder.create<mlir::dsa::Append>(
            loc,
            currentBuilder,
            value.getResult(),
            validValue.getResult()
        );
        
        ASSERT_NE(appendOp, nullptr) << "Failed to create append for standard width " << width;
        
        PGX_DEBUG(std::string("DSACharAppendTest: Created append for standard ") + std::to_string(width) + "-bit integer");
    }
    
    builder.create<mlir::func::ReturnOp>(loc);
    
    PGX_INFO("DSACharAppendTest: Successfully tested standard integer widths");
}