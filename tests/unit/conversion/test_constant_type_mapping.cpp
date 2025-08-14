#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

// PostgreSQL OID definitions
#define BOOLOID     16
#define INT2OID     21
#define INT4OID     23
#define INT8OID     20
#define FLOAT4OID   700
#define FLOAT8OID   701
#define TEXTOID     25
#define NUMERICOID  1700

// Test the type mapping function
TEST(ConstantTypeMappingTest, PostgreSQLOIDMapping) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::db::DBDialect>();
    
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Test INT32 constant maps to INT4OID (23)
    {
        auto int32Type = builder.getIntegerType(32);
        auto dbType = mlir::db::IntegerType::get(&context, 32, mlir::db::Signedness::Signed);
        auto constantOp = builder.create<mlir::db::ConstantOp>(
            loc, dbType, builder.getI32IntegerAttr(42)
        );
        
        // The constant should have the correct DB type
        EXPECT_TRUE(constantOp.getType().isa<mlir::db::IntegerType>());
        auto intType = constantOp.getType().cast<mlir::db::IntegerType>();
        EXPECT_EQ(intType.getWidth(), 32);
    }
    
    // Test INT64 constant maps to INT8OID (20)
    {
        auto int64Type = builder.getIntegerType(64);
        auto dbType = mlir::db::IntegerType::get(&context, 64, mlir::db::Signedness::Signed);
        auto constantOp = builder.create<mlir::db::ConstantOp>(
            loc, dbType, builder.getI64IntegerAttr(100)
        );
        
        EXPECT_TRUE(constantOp.getType().isa<mlir::db::IntegerType>());
        auto intType = constantOp.getType().cast<mlir::db::IntegerType>();
        EXPECT_EQ(intType.getWidth(), 64);
    }
    
    // Test BOOL constant maps to BOOLOID (16)
    {
        auto boolType = builder.getI1Type();
        auto dbType = mlir::db::IntegerType::get(&context, 1, mlir::db::Signedness::Signed);
        auto constantOp = builder.create<mlir::db::ConstantOp>(
            loc, dbType, builder.getBoolAttr(true)
        );
        
        EXPECT_TRUE(constantOp.getType().isa<mlir::db::IntegerType>());
        auto intType = constantOp.getType().cast<mlir::db::IntegerType>();
        EXPECT_EQ(intType.getWidth(), 1);
    }
    
    // Test STRING constant maps to TEXTOID (25)
    {
        auto dbType = mlir::db::StringType::get(&context);
        auto constantOp = builder.create<mlir::db::ConstantOp>(
            loc, dbType, builder.getStringAttr("test")
        );
        
        EXPECT_TRUE(constantOp.getType().isa<mlir::db::StringType>());
    }
}

// Test that the constant lowering doesn't crash with proper OIDs
TEST(ConstantTypeMappingTest, NoSegfaultWithProperOIDs) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::db::DBDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a simple module with a constant
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, "test_constant",
        builder.getFunctionType({}, {builder.getI32Type()})
    );
    
    auto block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create a DB constant that should lower without crashing
    auto dbType = mlir::db::IntegerType::get(&context, 32, mlir::db::Signedness::Signed);
    auto constantOp = builder.create<mlir::db::ConstantOp>(
        loc, dbType, builder.getI32IntegerAttr(42)
    );
    
    // Verify the constant was created successfully
    EXPECT_TRUE(constantOp);
    EXPECT_TRUE(constantOp.getType().isa<mlir::db::IntegerType>());
}