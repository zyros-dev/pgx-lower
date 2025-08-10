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
        context.loadDialect<func::FuncDialect, arith::ArithDialect,
                           pgx::db::DBDialect, pgx::mlir::dsa::DSADialect,
                           pgx::mlir::util::UtilDialect>();
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
    PGX_INFO("Testing DSAToStd with unconverted nullable type");
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Validate empty module
    ASSERT_TRUE(validateIR(module, "Empty module")) << "Empty module should be valid";
    
    // Create function with DSA operations and unconverted nullable types
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_unconverted", funcType);
    
    // Add entry block and temporary return BEFORE adding to module (required for verification)
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    auto tempReturn = builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    module.push_back(func);
    ASSERT_TRUE(validateIR(module, "After adding complete function")) << "Module with complete function should be valid";
    
    // Remove temporary return so we can add our operations
    tempReturn.erase();
    
    ASSERT_TRUE(validateIR(module, "After removing temp return")) << "Function ready for operations should be valid";
    
    // Create table builder
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    ASSERT_TRUE(validateIR(module, "After CreateDSOp")) << "Module should be valid after CreateDSOp";
    
    // Create a nullable value directly (simulating DBToStd not converting it)
    auto value = builder.create<arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 64);
    
    ASSERT_TRUE(validateIR(module, "After arith constant")) << "Module should be valid after constant";
    
    auto asNullableOp = builder.create<pgx::db::AsNullableOp>(
        builder.getUnknownLoc(), 
        pgx::db::NullableI64Type::get(&context),
        value.getResult());
    
    ASSERT_TRUE(validateIR(module, "After AsNullableOp")) << "Module should be valid after AsNullableOp";
    
    // Append the nullable value
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        asNullableOp.getResult());
    
    ASSERT_TRUE(validateIR(module, "After DSAppendOp")) << "Module should be valid after DSAppendOp";
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    ASSERT_TRUE(validateIR(module, "Complete IR before conversion")) << "Complete IR should be valid";
    
    // DEBUG: Test if the created IR is valid by walking it with depth limit
    PGX_INFO("Testing IR validity with bounded walk...");
    int walkDepth = 0;
    const int maxDepth = 100;
    bool hitDepthLimit = false;
    
    module.walk([&](Operation *op) -> WalkResult {
        walkDepth++;
        if (walkDepth > maxDepth) {
            PGX_ERROR("Walk depth exceeded " + std::to_string(maxDepth) + " - circular IR detected!");
            hitDepthLimit = true;
            return WalkResult::interrupt();
        }
        
        // Log every operation to see the circular pattern
        std::string opInfo = "Depth " + std::to_string(walkDepth) + ": " + op->getName().getStringRef().str();
        if (op->getNumResults() > 0) {
            opInfo += " (results: " + std::to_string(op->getNumResults()) + ")";
        }
        if (op->getNumOperands() > 0) {
            opInfo += " (operands: " + std::to_string(op->getNumOperands()) + ")";
        }
        if (op->getNumRegions() > 0) {
            opInfo += " (regions: " + std::to_string(op->getNumRegions()) + ")";
        }
        
        PGX_INFO(opInfo);
        
        return WalkResult::advance();
    });
    
    PGX_INFO("Walk completed at depth: " + std::to_string(walkDepth));
    
    if (hitDepthLimit) {
        FAIL() << "Test IR has circular references - depth exceeded " << maxDepth;
        return;
    }
    
    // Try a simple dump with timeout protection
    PGX_INFO("Testing module dump...");
    // module.dump(); // Skip this for now since it causes infinite loops
    
    // Run only DSAToStd pass (without DBToStd)
    PassManager pm(&context);
    pm.addPass(::mlir::createDSAToStdPass());
    
    // This should succeed without circular materialization
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the nullable type was handled correctly
    bool foundNullableAppend = false;
    module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == "pgx_runtime_append_nullable_i64") {
            foundNullableAppend = true;
            PGX_DEBUG("Found nullable append call after DSAToStd");
        }
    });
    
    EXPECT_TRUE(foundNullableAppend) << "DSAToStd should handle unconverted nullable types";
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
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
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
    builder.create<pgx::mlir::dsa::DSAppendOp>(
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
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
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
    builder.create<pgx::mlir::dsa::DSAppendOp>(
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
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    ASSERT_TRUE(validateIR(module, "After DSA CreateDSOp")) 
        << "Module should be valid after CreateDSOp";
    
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
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(),
        createDSOp.getResult(),
        asNullableOp.getResult());
    
    ASSERT_TRUE(validateIR(module, "After DSA DSAppendOp")) 
        << "Module should be valid after DSAppendOp";
    
    // Step 8: Add return and final validation
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    ASSERT_TRUE(validateIR(module, "Complete IR before conversion")) 
        << "Complete IR should be valid before conversion passes";
    
    PGX_INFO("✓ All IR construction steps validated successfully");
    
    // Optional: Print the IR to see what we built
    PGX_INFO("=== Generated IR ===");
    module.print(llvm::errs());
}