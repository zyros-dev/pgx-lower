//===- test_db_to_std_basic.cpp - Basic DB to Standard conversion tests ---===//

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBTypes.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSADialect.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h"
#include "pgx_lower/mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"

using namespace mlir;

class DBToStdBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<mlir::db::DBDialect>();
        context.getOrLoadDialect<mlir::dsa::DSADialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<memref::MemRefDialect>();
    }
    
    MLIRContext context;
};

TEST_F(DBToStdBasicTest, ConvertGetExternalToSPICall) {
    PGX_INFO("Testing db.get_external to pg_table_open conversion");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function with db.get_external operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create db.get_external operation
    auto tableOid = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 16384, 64);
    auto getExternalOp = builder.create<mlir::db::GetExternalOp>(
        builder.getUnknownLoc(), tableOid.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the conversion
    bool foundSPICall = false;
    module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == "pg_table_open") {
            foundSPICall = true;
            EXPECT_EQ(callOp.getNumOperands(), 1);
            EXPECT_TRUE(callOp.getOperand(0).getType().isInteger(64));
        }
    });
    
    EXPECT_TRUE(foundSPICall) << "Expected pg_table_open SPI call not found";
}

TEST_F(DBToStdBasicTest, ConvertIterateExternalToSPICall) {
    PGX_INFO("Testing db.iterate_external to pg_get_next_tuple conversion");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function with db.iterate_external operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create db.get_external first
    auto tableOid = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 16384, 64);
    auto getExternalOp = builder.create<mlir::db::GetExternalOp>(
        builder.getUnknownLoc(), tableOid.getResult());
    
    // Create db.iterate_external
    auto iterateOp = builder.create<mlir::db::IterateExternalOp>(
        builder.getUnknownLoc(), builder.getI1Type(), 
        getExternalOp.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the conversion
    bool foundIterateCall = false;
    module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == "pg_get_next_tuple") {
            foundIterateCall = true;
            EXPECT_EQ(callOp.getNumOperands(), 1);
            EXPECT_TRUE(callOp.getResult(0).getType().isInteger(1));
        }
    });
    
    EXPECT_TRUE(foundIterateCall) << "Expected pg_get_next_tuple SPI call not found";
}

TEST_F(DBToStdBasicTest, ConvertAddToArithAddi) {
    PGX_INFO("Testing db.add to arith.addi conversion");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function with db.add operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create db.add operation
    auto lhs = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 5, 64);
    auto rhs = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 10, 64);
    auto addOp = builder.create<mlir::db::AddOp>(
        builder.getUnknownLoc(), builder.getI64Type(), lhs.getResult(), rhs.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the conversion
    bool foundArithAdd = false;
    module.walk([&](arith::AddIOp addOp) {
        foundArithAdd = true;
        EXPECT_TRUE(addOp.getType().isInteger(64));
    });
    
    EXPECT_TRUE(foundArithAdd) << "Expected arith.addi operation not found";
}

// TEMPORARILY DISABLED: DSA operations removed in Phase 4d architecture cleanup
/*
TEST_F(DBToStdBasicTest, PreserveDSAOperations) {
    PGX_INFO("Testing that DSA operations are preserved during conversion");
    
    // This test has been disabled as we've removed the DSA data structure
    // building operations (CreateDS, Append) in favor of using 
    // DB operations directly.
    
    PGX_INFO("DSA operations test skipped - operations removed");
}
*/

TEST_F(DBToStdBasicTest, GenerateSPIFunctionDeclarations) {
    PGX_INFO("Testing SPI function declaration generation");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function that uses DB operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create operations that will need SPI functions
    auto tableOid = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 16384, 64);
    auto getExternalOp = builder.create<mlir::db::GetExternalOp>(
        builder.getUnknownLoc(), tableOid.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify SPI function declarations exist
    auto pgTableOpen = module.lookupSymbol<func::FuncOp>("pg_table_open");
    ASSERT_TRUE(pgTableOpen) << "pg_table_open function declaration not found";
    
    auto pgGetNextTuple = module.lookupSymbol<func::FuncOp>("pg_get_next_tuple");
    ASSERT_TRUE(pgGetNextTuple) << "pg_get_next_tuple function declaration not found";
    
    auto pgExtractField = module.lookupSymbol<func::FuncOp>("pg_extract_field");
    ASSERT_TRUE(pgExtractField) << "pg_extract_field function declaration not found";
}

TEST_F(DBToStdBasicTest, ConvertStoreResultToSPICall) {
    PGX_INFO("Testing db.store_result to pg_store_result conversion");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function that accepts a nullable value as parameter
    auto nullableType = mlir::db::NullableI32Type::get(&context);
    auto funcType = builder.getFunctionType({nullableType}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Get the function argument (nullable value)
    auto nullableValue = block->getArgument(0);
    
    // Create field index
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 0);
    
    // Create db.store_result operation
    builder.create<mlir::db::StoreResultOp>(
        builder.getUnknownLoc(), 
        nullableValue, 
        fieldIndex.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the conversion
    bool foundStoreCall = false;
    module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee().starts_with("pg_store_result")) {
            foundStoreCall = true;
            EXPECT_EQ(callOp.getNumOperands(), 2);
            // First operand should be a memref (to the nullable tuple)
            auto memrefType = callOp.getOperand(0).getType().dyn_cast<MemRefType>();
            EXPECT_TRUE(memrefType) << "Expected memref type for nullable value";
            // Verify it's a memref to a tuple type
            auto elementType = memrefType.getElementType().dyn_cast<TupleType>();
            EXPECT_TRUE(elementType) << "Expected memref to tuple type";
            // Second operand should be index type
            EXPECT_TRUE(callOp.getOperand(1).getType().isIndex());
        }
    });
    
    EXPECT_TRUE(foundStoreCall) << "Expected pg_store_result SPI call not found";
    
    // Verify SPI function declaration exists (check for type-specific version)
    auto pgStoreResult = module.lookupSymbol<func::FuncOp>("pg_store_result_i32");
    ASSERT_TRUE(pgStoreResult) << "pg_store_result_i32 function declaration not found";
}