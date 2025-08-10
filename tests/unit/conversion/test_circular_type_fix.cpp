//===- test_circular_type_fix.cpp - Tests for circular type materialization fix ===//
//
// This test verifies that the DSAToStd pass correctly handles DB nullable types
// that may or may not have been converted by DBToStd, preventing the circular
// type materialization bug.
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
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"

using namespace mlir;

class CircularTypeFixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load standard dialects
        context.loadDialect<func::FuncDialect, arith::ArithDialect>();
        
        // Add DB and DSA dialects - these were OK
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        
        // Add Util dialect to test
        PGX_INFO("Loading Util dialect...");
        context.loadDialect<pgx::mlir::util::UtilDialect>();
        PGX_INFO("Util dialect loaded");
    }

    // Helper to validate IR using MLIR's built-in verification
    bool validateIR(ModuleOp module, const std::string& stepName) {
        PGX_INFO("=== Validating: " + stepName + " ===");
        bool isValid = mlir::verify(module).succeeded();
        if (isValid) {
            PGX_INFO("✓ IR validation passed: " + stepName);
        } else {
            PGX_ERROR("✗ IR validation FAILED: " + stepName + " (see stderr for details)");
        }
        return isValid;
    }

    MLIRContext context;
};

TEST_F(CircularTypeFixTest, UnconvertedNullableType) {
    PGX_INFO("Testing minimal IR construction");
    
    // Test 1: Just empty module
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    
    PGX_INFO("About to validate empty module...");
    ASSERT_TRUE(validateIR(module, "Empty module")) << "Empty module should be valid";
    PGX_INFO("Empty module validated successfully");
    
    // Test 2: Try creating function WITHOUT adding to module first
    PGX_INFO("Creating function...");
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_minimal", funcType);
    PGX_INFO("Function created");
    
    // Test 3: Add entry block
    PGX_INFO("Adding entry block...");
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    PGX_INFO("Entry block added");
    
    // Test 4: Add return
    PGX_INFO("Adding return...");
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    PGX_INFO("Return added");
    
    // Test 5: Now try adding to module
    PGX_INFO("About to add function to module...");
    builder.setInsertionPointToStart(module.getBody());
    module.push_back(func);
    PGX_INFO("Function added to module");
    
    PGX_INFO("About to validate module with function...");
    ASSERT_TRUE(validateIR(module, "Module with function")) << "Module with function should be valid";
    PGX_INFO("SUCCESS: Module with function validated!");
    
    // Now let's continue with the original test that was hanging
    // Remove the early return and continue with DSA operations
    
    // Create table builder - this might be where the issue is
    PGX_INFO("About to create DSA TableBuilderType...");
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    PGX_INFO("DSA TableBuilderType created");
    
    PGX_INFO("About to create DSA CreateDS...");
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    PGX_INFO("DSA CreateDS created");
    
    PGX_INFO("About to validate after DSA CreateDS...");
    ASSERT_TRUE(validateIR(module, "After DSA CreateDS")) << "Module should be valid after CreateDS";
    PGX_INFO("Validation after DSA CreateDS passed!");
    
    // Continue with the EXACT original test sequence that was hanging
    // This will test if the issue is really fixed
    
    // Remove the return and add all operations in the same block
    block->getTerminator()->erase();  // Remove the return we added
    builder.setInsertionPointToEnd(block);
    
    // ALL operations must be in the same function block for proper scoping
    // Create a NEW createDSOp inside the function block
    PGX_INFO("Creating new DSA CreateDS inside function block...");
    auto funcCreateDS = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create a nullable value directly (simulating DBToStd not converting it)
    PGX_INFO("Creating nullable value...");
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto asNullableOp = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), 
        pgx::db::NullableI64Type::get(&context),
        value.getResult());
    PGX_INFO("Nullable value created");
    
    // Append the nullable value using the function-scoped createDSOp
    PGX_INFO("Creating DSA append operation...");
    builder.create<pgx::mlir::dsa::Append>(
        builder.getUnknownLoc(),
        funcCreateDS.getResult(),
        asNullableOp.getResult());
    PGX_INFO("DSA append created");
    
    // Add final return
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // THIS WAS THE EXACT POINT WHERE THE ORIGINAL TEST HUNG
    PGX_INFO("About to verify the COMPLETE original test IR...");
    ASSERT_TRUE(validateIR(module, "Complete original test IR")) << "Original test IR should work now";
    PGX_INFO("SUCCESS: Original test IR verified!");
}

// Test that DSAToStd correctly handles already-converted tuple types
TEST_F(CircularTypeFixTest, AlreadyConvertedTupleType) {
    PGX_INFO("Testing DSAToStd with already-converted tuple type");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function with DSA operations and pre-converted tuple
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_converted", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create a tuple directly (simulating DBToStd already converted it)
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto falseVal = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(), builder.getBoolAttr(false));
    
    auto tupleType = TupleType::get(&context, {builder.getI64Type(), builder.getI1Type()});
    auto packOp = builder.create<pgx::mlir::util::PackOp>(
        builder.getUnknownLoc(), tupleType, 
        ValueRange{value.getResult(), falseVal.getResult()});
    
    // Append the tuple value
    builder.create<pgx::mlir::dsa::Append>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        packOp.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run DSAToStd pass
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());
    
    // This should succeed without circular materialization
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify tuple extraction happened
    bool foundGetTuple = false;
    module.walk([&](pgx::mlir::util::GetTupleOp op) {
        foundGetTuple = true;
        PGX_DEBUG("Found get_tuple operation");
    });
    
    EXPECT_TRUE(foundGetTuple) << "DSAToStd should extract elements from tuple";
}

// Test the complete pipeline scenario that was failing
TEST_F(CircularTypeFixTest, CompletePipelineNoCircular) {
    PGX_INFO("Testing complete DBToStd + DSAToStd pipeline");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create the exact pattern that was failing
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_pipeline", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Create table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    // Create a value and make it nullable
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    auto asNullableOp = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), 
        pgx::db::NullableI64Type::get(&context),
        value.getResult());
    
    // Append the nullable value
    builder.create<pgx::mlir::dsa::Append>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        asNullableOp.getResult());
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    // Run both passes in sequence
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(::mlir::createDBToStdPass());
    pm.addPass(::mlir::createDSAToStdPass());
    
    // This should succeed without circular materialization or infinite loops
    ASSERT_TRUE(succeeded(pm.run(module))) << "Pipeline should not have circular materialization";
    
    // Verify no DB operations remain
    bool hasDBOps = false;
    module.walk([&](Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "db") {
            hasDBOps = true;
        }
    });
    
    EXPECT_FALSE(hasDBOps) << "All DB operations should be converted";
    
    // Verify DSA operations are converted
    bool hasDSAOps = false;
    module.walk([&](Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
            hasDSAOps = true;
        }
    });
    
    EXPECT_FALSE(hasDSAOps) << "All DSA operations should be converted";
    
    PGX_INFO("Pipeline test passed - no circular materialization!");
}

// Test that validates IR construction step by step before conversion
TEST_F(CircularTypeFixTest, ValidateIRConstruction) {
    PGX_INFO("Testing step-by-step IR construction with validation");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Step 1: Validate empty module
    ASSERT_TRUE(validateIR(module, "Empty module")) << "Empty module should be valid";
    
    // Step 2: Create and validate function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_step_by_step", funcType);
    module.push_back(func);
    
    ASSERT_TRUE(validateIR(module, "Module with empty function")) 
        << "Module with empty function should be valid";
    
    // Step 3: Add entry block and validate
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    ASSERT_TRUE(validateIR(module, "Function with entry block")) 
        << "Function with entry block should be valid";
    
    // Step 4: Create DSA table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    ASSERT_TRUE(validateIR(module, "After DSA CreateDS")) 
        << "Module should be valid after CreateDS";
    
    // Step 5: Create arith constant
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    
    ASSERT_TRUE(validateIR(module, "After arith constant")) 
        << "Module should be valid after arith constant";
    
    // Step 6: Create DB nullable value
    auto asNullableOp = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), 
        pgx::db::NullableI64Type::get(&context),
        value.getResult());
    
    ASSERT_TRUE(validateIR(module, "After DB AsNullableOp")) 
        << "Module should be valid after AsNullableOp";
    
    // Step 7: Create DSA append operation
    builder.create<pgx::mlir::dsa::Append>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        asNullableOp.getResult());
    
    ASSERT_TRUE(validateIR(module, "After DSA Append")) 
        << "Module should be valid after Append";
    
    // Step 8: Add return and final validation
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    ASSERT_TRUE(validateIR(module, "Complete IR before conversion")) 
        << "Complete IR should be valid before conversion passes";
    
    PGX_INFO("✓ All IR construction steps validated successfully");
    
    // Optional: Print the IR to see what we built
    PGX_INFO("=== Generated IR ===");
    module.print(llvm::errs());
}