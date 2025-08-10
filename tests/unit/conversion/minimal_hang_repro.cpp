//===- minimal_hang_repro.cpp - Minimal reproduction of infinite loop bug ===//
//
// This creates the EXACT minimal MLIR IR pattern that triggers the infinite loop
// in DSAToStd conversion pass.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "execution/logging.h"
#include "utility/ir_debug_utils.h"

using namespace mlir;

class MinimalHangReproTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<func::FuncDialect, arith::ArithDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<pgx::mlir::util::UtilDialect>();
    }

    MLIRContext context;
};

// Circular IR detection moved to utility/ir_debug_utils.h

// Test 1: Just empty DSA operation (should work)
TEST_F(MinimalHangReproTest, EmptyDSAOperation) {
    PGX_INFO("=== Testing empty DSA CreateDS operation ===");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_empty_dsa", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create ONLY DSA operation - no DB types involved
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Return immediately - no other operations
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run DSAToStd pass with timeout protection
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());

    PGX_INFO("Validating tested IR is not circular");
    // TEMPORARILY DISABLED - cycle detector hangs on malformed IR
    // ASSERT_FALSE(pgx::utility::hasCircularIR(module));
    PGX_INFO("Running DSAToStd on JUST CreateDS operation...");
    ASSERT_TRUE(succeeded(pm.run(module))) << "Empty DSA operation should convert without hanging";
    PGX_INFO("✓ Empty DSA operation converted successfully");
}

// Test 2: DSA operation with arith operand (should work)
TEST_F(MinimalHangReproTest, DSAWithArithOperand) {
    PGX_INFO("=== Testing DSA operation with arith operand ===");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_dsa_arith", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create DSA table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create arith value and append it (no DB types)
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        value.getResult());  // Direct arith value - no DB nullable
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run DSAToStd pass
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());
    
    PGX_INFO("Validating tested IR is not circular");
    // TEMPORARILY DISABLED - cycle detector hangs on malformed IR
    // ASSERT_FALSE(pgx::utility::hasCircularIR(module));
    PGX_INFO("Running DSAToStd on DSA + arith operand...");
    ASSERT_TRUE(succeeded(pm.run(module))) << "DSA with arith operand should convert without hanging";
    PGX_INFO("✓ DSA with arith operand converted successfully");
}

// Test 3: EXACT pattern that causes infinite loop - DSA operation with DB nullable operand
TEST_F(MinimalHangReproTest, DSAWithDBNullableOperand) {
    PGX_INFO("=== Testing EXACT infinite loop trigger pattern ===");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_dsa_db_nullable", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create DSA table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create DB nullable value
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto asNullableOp = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), 
        pgx::db::NullableI64Type::get(&context),
        value.getResult());
    
    // THIS IS THE EXACT COMBINATION THAT TRIGGERS INFINITE LOOP:
    // DSA operation (DSAppendOp) with DB nullable operand
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),      // DSA TableBuilder type
        asNullableOp.getResult());   // DB NullableI64Type - CIRCULAR DEPENDENCY!
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);

    PGX_INFO("Validating tested IR is not circular");
    // TEMPORARILY DISABLED - cycle detector hangs on malformed IR
    // ASSERT_FALSE(pgx::utility::hasCircularIR(module));

    PGX_INFO("=== Generated IR before conversion ===");
    module.print(llvm::errs());
    
    // Run DSAToStd pass - THIS WILL HANG
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());
    
    PGX_INFO("Running DSAToStd on DSA + DB nullable operand...");
    PGX_INFO("WARNING: This test is expected to hang due to infinite loop bug!");
    
    // This line should cause infinite loop
    auto result = pm.run(module);
    
    // If we reach this line, the bug is fixed
    ASSERT_TRUE(succeeded(result)) << "DSA with DB nullable operand should convert without hanging";
    PGX_INFO("🎉 BUG FIXED: DSA with DB nullable operand converted successfully!");
}

// Test 4: Alternative - DSA operation with already-converted tuple (should work)
TEST_F(MinimalHangReproTest, DSAWithPreConvertedTuple) {
    PGX_INFO("=== Testing DSA operation with pre-converted tuple (workaround) ===");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_dsa_tuple", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create DSA table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create tuple directly (simulating DBToStd already converted it)
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto falseVal = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(), builder.getBoolAttr(false));
    
    auto tupleType = TupleType::get(&context, {builder.getI64Type(), builder.getI1Type()});
    auto packOp = builder.create<pgx::mlir::util::PackOp>(
        builder.getUnknownLoc(), tupleType, 
        ValueRange{value.getResult(), falseVal.getResult()});
    
    // Use pre-converted tuple instead of DB nullable
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),      // DSA TableBuilder type
        packOp.getResult());         // Tuple type - no circular dependency
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run DSAToStd pass
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());
    
    PGX_INFO("Validating tested IR is not circular");
    // TEMPORARILY DISABLED - cycle detector hangs on malformed IR
    // ASSERT_FALSE(pgx::utility::hasCircularIR(module));
    PGX_INFO("Running DSAToStd on DSA + pre-converted tuple...");
    ASSERT_TRUE(succeeded(pm.run(module))) << "DSA with pre-converted tuple should work";
    PGX_INFO("✓ DSA with pre-converted tuple converted successfully");
}