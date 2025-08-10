//===- test_dbtostd_edge_cases.cpp - DBToStd Edge Case Tests ----===//
//
// Tests for edge cases in the DBToStd conversion pass to ensure
// robustness and proper error handling.
//
//===-------------------------------------------------------------===//

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "execution/logging.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arith;
using namespace mlir::scf;

class DBToStdEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<FuncDialect, ArithDialect, SCFDialect, 
                           memref::MemRefDialect, pgx::db::DBDialect,
                           pgx::mlir::dsa::DSADialect, 
                           pgx::mlir::util::UtilDialect>();
    }

    MLIRContext context;
};

// Test empty function handling
TEST_F(DBToStdEdgeCaseTest, EmptyFunctionHandling) {
    PGX_INFO("Testing DBToStd pass on empty function");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create an empty function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "empty_func", funcType);
    func.addEntryBlock();
    
    // Add return to make it valid
    builder.setInsertionPointToEnd(&func.getBody().front());
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(createDBToStdPass());
    
    // Should succeed on empty function
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    PGX_INFO("Empty function test passed");
}

// Test function with only non-DB operations
TEST_F(DBToStdEdgeCaseTest, NonDBOperationsOnly) {
    PGX_INFO("Testing DBToStd pass on function with only arithmetic operations");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function with only arithmetic operations
    auto funcType = builder.getFunctionType({builder.getI64Type()}, 
                                           {builder.getI64Type()});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "arith_only_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Add some arithmetic operations
    auto arg = block->getArgument(0);
    auto constant = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto add = builder.create<arith::AddIOp>(
        builder.getUnknownLoc(), arg, constant.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), add.getResult());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(createDBToStdPass());
    
    // Should succeed and preserve arithmetic operations
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify arithmetic operations are preserved
    bool foundAdd = false;
    module.walk([&](arith::AddIOp) { foundAdd = true; });
    EXPECT_TRUE(foundAdd) << "Arithmetic operations should be preserved";
    
    PGX_INFO("Non-DB operations test passed");
}

// Test mixed DB and non-DB operations
TEST_F(DBToStdEdgeCaseTest, MixedOperations) {
    PGX_INFO("Testing DBToStd pass on function with mixed operations");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function with mixed operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "mixed_ops_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Add arithmetic operation
    auto constant1 = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 10, 64);
    auto constant2 = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 20, 64);
    auto add = builder.create<arith::AddIOp>(
        builder.getUnknownLoc(), constant1.getResult(), constant2.getResult());
    
    // Add DB operation
    auto dbAdd = builder.create<pgx::db::AddOp>(
        builder.getUnknownLoc(), builder.getI64Type(),
        constant1.getResult(), constant2.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(createDBToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify DB operations are converted
    int dbOpCount = 0;
    module.walk([&](Operation* op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "db") {
            dbOpCount++;
        }
    });
    EXPECT_EQ(dbOpCount, 0) << "All DB operations should be converted";
    
    // Verify arithmetic operations exist (both original and converted)
    int arithAddCount = 0;
    module.walk([&](arith::AddIOp) { arithAddCount++; });
    EXPECT_GE(arithAddCount, 2) << "Should have original and converted add operations";
    
    PGX_INFO("Mixed operations test passed");
}

// Test proper type conversion verification
TEST_F(DBToStdEdgeCaseTest, ProperNullableTypeHandling) {
    PGX_INFO("Testing proper nullable type conversion");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function that properly uses nullable types
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "nullable_test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Get external table
    auto tableOid = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 16384, 64);
    auto getExternalOp = builder.create<pgx::db::GetExternalOp>(
        builder.getUnknownLoc(), tableOid.getResult());
    
    // Get field (returns nullable type)
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 0);
    auto typeOid = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 20, 32);
    
    auto getFieldOp = builder.create<pgx::db::GetFieldOp>(
        builder.getUnknownLoc(), 
        builder.getType<pgx::db::NullableI64Type>(),
        getExternalOp.getResult(),
        fieldIndex.getResult(),
        typeOid.getResult());
    
    // Extract value from nullable
    auto getValOp = builder.create<pgx::db::NullableGetValOp>(
        builder.getUnknownLoc(), builder.getI64Type(), 
        getFieldOp.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run the conversion pass
    PassManager pm(&context);
    pm.addPass(createDBToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify nullable_get_val is converted to util.get_tuple
    bool foundGetTuple = false;
    module.walk([&](Operation* op) { 
        if (op->getName().getStringRef() == "util.get_tuple") {
            foundGetTuple = true;
            // Verify it's extracting from index 0 (the value)
            if (auto getTupleOp = dyn_cast<pgx::mlir::util::GetTupleOp>(op)) {
                EXPECT_EQ(getTupleOp.getOffset(), 0) << "Should extract value at index 0";
            }
        }
    });
    EXPECT_TRUE(foundGetTuple) << "nullable_get_val should convert to util.get_tuple";
    
    PGX_INFO("Nullable type handling test passed");
}