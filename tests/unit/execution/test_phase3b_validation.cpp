#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Passes.h"
#include "execution/logging.h"

using namespace mlir;

class Phase3bValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<MLIRContext>();
        
        // Load all required dialects
        context->getOrLoadDialect<relalg::RelAlgDialect>();
        context->getOrLoadDialect<db::DBDialect>();
        context->getOrLoadDialect<dsa::DSADialect>();
        context->getOrLoadDialect<util::UtilDialect>();
        context->getOrLoadDialect<func::FuncDialect>();
    }
    
    std::unique_ptr<MLIRContext> context;
};

// Test that module validation catches malformed MLIR
TEST_F(Phase3bValidationTest, ModuleValidationCatchesMalformedIR) {
    // Create a module
    auto module = ModuleOp::create(UnknownLoc::get(context.get()));
    
    // Create a function with invalid structure (missing terminator)
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto func = func::FuncOp::create(UnknownLoc::get(context.get()), "test", funcType);
    func.addEntryBlock();
    // Deliberately don't add a terminator to make it invalid
    
    module.push_back(func);
    
    // Verification should fail
    EXPECT_TRUE(failed(verify(module.getOperation())));
}

// Test that dialect verification works
TEST_F(Phase3bValidationTest, DialectVerificationWorks) {
    // Verify all dialects are loaded
    auto* dbDialect = context->getLoadedDialect<db::DBDialect>();
    auto* dsaDialect = context->getLoadedDialect<dsa::DSADialect>();
    
    EXPECT_NE(dbDialect, nullptr);
    EXPECT_NE(dsaDialect, nullptr);
}

// Test sequential pass execution with canonicalization
TEST_F(Phase3bValidationTest, SequentialPassExecutionWithCanonicalization) {
    // Create a valid module
    auto module = ModuleOp::create(UnknownLoc::get(context.get()));
    
    // Create a simple function
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto func = func::FuncOp::create(UnknownLoc::get(context.get()), "test", funcType);
    auto* entryBlock = func.addEntryBlock();
    
    // Add a return operation to make it valid
    OpBuilder builder(entryBlock, entryBlock->begin());
    builder.create<func::ReturnOp>(UnknownLoc::get(context.get()));
    
    module.push_back(func);
    
    // Verify initial module
    EXPECT_TRUE(succeeded(verify(module.getOperation())));
    
    // Create pass manager with DB+DSA→Standard pipeline
    PassManager pm(context.get());
    pgx_lower::createDBDSAToStandardPipeline(pm, true);
    
    // Pipeline should run successfully on valid module
    EXPECT_TRUE(succeeded(pm.run(module)));
    
    // Module should still be valid after lowering
    EXPECT_TRUE(succeeded(verify(module.getOperation())));
}

// Test IR dumping functionality
TEST_F(Phase3bValidationTest, IRDumpingWorks) {
    // Create a module
    auto module = ModuleOp::create(UnknownLoc::get(context.get()));
    
    // Test that we can dump IR to string
    std::string moduleStr;
    llvm::raw_string_ostream os(moduleStr);
    module.getOperation()->print(os);
    
    // Should contain module markers
    EXPECT_NE(moduleStr.find("module"), std::string::npos);
}

// Test logging level checks
TEST_F(Phase3bValidationTest, LoggingLevelChecks) {
    // Test that logging level check works
    auto& logger = get_logger();
    
    // Save current level
    auto originalLevel = logger.should_log(LogLevel::DEBUG_LVL);
    
    // Set to INFO level
    logger.set_level(LogLevel::INFO_LVL);
    
    // DEBUG should not be logged
    EXPECT_FALSE(logger.should_log(LogLevel::DEBUG_LVL));
    
    // INFO should be logged
    EXPECT_TRUE(logger.should_log(LogLevel::INFO_LVL));
    
    // Restore original level
    if (originalLevel) {
        logger.set_level(LogLevel::DEBUG_LVL);
    }
}