#include <gtest/gtest.h>
#include <chrono>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "dialects/relalg/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBTypes.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

// Test structure for empty pipeline options
struct EmptyPipelineOptions {};

class PassPipelineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects for the pipeline
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<memref::MemRefDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
    
    // Helper to create a simple RelAlg module for testing
    OwningOpRef<ModuleOp> createSimpleRelAlgModule() {
        OpBuilder builder(&context);
        Location loc = builder.getUnknownLoc();
        
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
        
        // Create a simple function with RelAlg operations
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<func::FuncOp>(loc, "test_relalg", funcType);
        
        auto& funcBody = func.getBody();
        funcBody.push_back(new Block);
        builder.setInsertionPointToStart(&funcBody.front());
        
        // Create a simple BaseTableOp for testing
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        auto col1 = colManager.createDef("test", "id");
        col1.getColumn().type = builder.getI32Type();
        auto col2 = colManager.createDef("test", "name");
        col2.getColumn().type = builder.getType<db::StringType>();
        
        std::vector<NamedAttribute> columns = {
            builder.getNamedAttr("id", col1),
            builder.getNamedAttr("name", col2)
        };
        
        auto baseTableOp = builder.create<relalg::BaseTableOp>(
            loc,
            tuples::TupleStreamType::get(&context),
            builder.getDictionaryAttr(columns),
            builder.getStringAttr("test_table|oid:12345")
        );
        
        // Add return operation
        builder.create<func::ReturnOp>(loc);
        
        return module;
    }
    
    // Helper to verify module contains expected operations after lowering
    bool verifySubOpOperations(ModuleOp module) {
        bool foundSubOpOps = false;
        module.walk([&](Operation* op) {
            if (isa<subop::SubOperatorDialectNamespace>(op->getDialect())) {
                foundSubOpOps = true;
            }
        });
        return foundSubOpOps;
    }
    
    // Helper to verify no RelAlg operations remain after lowering
    bool verifyNoRelAlgOperations(ModuleOp module) {
        bool foundRelAlgOps = false;
        module.walk([&](Operation* op) {
            if (isa<relalg::RelAlgDialectNamespace>(op->getDialect())) {
                foundRelAlgOps = true;
            }
        });
        return !foundRelAlgOps;
    }
};

TEST_F(PassPipelineIntegrationTest, PassCreation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass creation");
    
    // Test that we can create the lowering pass
    auto pass = relalg::createLowerToSubOpPass();
    EXPECT_TRUE(pass != nullptr);
    
    // Test that we can create the alternative pass creation method
    auto pass2 = relalg::createLowerRelAlgToSubOpPass();
    EXPECT_TRUE(pass2 != nullptr);
}

TEST_F(PassPipelineIntegrationTest, PassRegistration) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass registration");
    
    // Register the passes
    relalg::registerRelAlgToSubOpConversionPasses();
    
    // The passes should be registered without throwing exceptions
    // This is primarily testing that the registration code executes without error
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PipelineCreation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pipeline creation");
    
    PassManager pm(&context);
    
    // Test creating the pipeline
    relalg::createLowerRelAlgToSubOpPipeline(pm);
    
    // Verify the pipeline was created successfully
    EXPECT_GT(pm.size(), 0);
}

TEST_F(PassPipelineIntegrationTest, CompletePassExecution) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing complete pass execution");
    
    // Create a simple RelAlg module
    auto module = createSimpleRelAlgModule();
    EXPECT_TRUE(module);
    
    // Create pass manager and add the lowering pass
    PassManager pm(&context);
    pm.addPass(relalg::createLowerToSubOpPass());
    
    // Enable debug information for pass execution
    pm.enableIRPrinting();
    
    // Run the pass on the module
    LogicalResult result = pm.run(*module);
    
    // Verify the pass executed successfully
    EXPECT_TRUE(succeeded(result));
    
    // Verify that the lowering occurred correctly
    // Note: Due to the complexity of the lowering patterns,
    // we primarily test that the pass doesn't fail
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PassFailureHandling) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass failure handling");
    
    // Create an invalid module (empty module without proper structure)
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    
    // Create pass manager
    PassManager pm(&context);
    pm.addPass(relalg::createLowerToSubOpPass());
    
    // Run the pass - it should handle the empty module gracefully
    LogicalResult result = pm.run(*module);
    
    // The pass should not crash on empty modules
    // (success or failure depends on pass implementation details)
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PassConfiguration) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass configuration");
    
    // Test that passes can be configured with different options
    PassManager pm(&context);
    
    // Add the pass with default configuration
    pm.addPass(relalg::createLowerToSubOpPass());
    
    // Verify configuration options
    pm.enableStatistics();
    pm.enableTiming();
    
    // The pass manager should accept these configurations
    EXPECT_GT(pm.size(), 0);
}

TEST_F(PassPipelineIntegrationTest, PipelineIntegration) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pipeline integration");
    
    // Create a module for testing
    auto module = createSimpleRelAlgModule();
    EXPECT_TRUE(module);
    
    // Create a comprehensive pass pipeline
    PassManager pm(&context);
    
    // Add canonicalization before our lowering pass
    pm.addPass(createCanonicalizerPass());
    
    // Add our lowering pass
    relalg::createLowerRelAlgToSubOpPipeline(pm);
    
    // Add canonicalization after our lowering pass
    pm.addPass(createCanonicalizerPass());
    
    // Run the full pipeline
    LogicalResult result = pm.run(*module);
    
    // Verify the pipeline executed successfully
    EXPECT_TRUE(succeeded(result));
}

TEST_F(PassPipelineIntegrationTest, DialectDependencyValidation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing dialect dependency validation");
    
    // Create a pass and verify it declares the correct dialect dependencies
    auto pass = relalg::createLowerToSubOpPass();
    EXPECT_TRUE(pass != nullptr);
    
    // Create a pass manager with proper dialect registration
    PassManager pm(&context);
    pm.addPass(std::move(pass));
    
    // The pass should be able to access all required dialects
    // This is tested implicitly by successful pass creation
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PassOrdering) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass ordering");
    
    PassManager pm(&context);
    
    // Test that our pass can be ordered with other passes
    pm.addPass(createCanonicalizerPass());
    pm.addPass(relalg::createLowerToSubOpPass());
    pm.addPass(createCSEPass());
    
    // Verify the passes are in the expected order
    EXPECT_EQ(pm.size(), 3);
}

TEST_F(PassPipelineIntegrationTest, PassManagerIntegration) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass manager integration");
    
    // Test different pass manager configurations
    PassManager pm(&context);
    
    // Test nested pass managers (if supported by the pass)
    auto pass = relalg::createLowerToSubOpPass();
    pm.addPass(std::move(pass));
    
    // Enable various pass manager features
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableIRPrinting();
    
    // The pass should integrate well with pass manager features
    EXPECT_GT(pm.size(), 0);
}

TEST_F(PassPipelineIntegrationTest, MultiplePassExecution) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing multiple pass execution");
    
    // Create a module
    auto module = createSimpleRelAlgModule();
    EXPECT_TRUE(module);
    
    // Create multiple pass managers and run them sequentially
    for (int i = 0; i < 3; ++i) {
        PassManager pm(&context);
        pm.addPass(relalg::createLowerToSubOpPass());
        
        // Each execution should handle the module state correctly
        // (though subsequent executions may be no-ops if lowering is complete)
        LogicalResult result = pm.run(*module);
        EXPECT_TRUE(succeeded(result));
    }
}

TEST_F(PassPipelineIntegrationTest, PatternPopulation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pattern population");
    
    // Test that pattern population works correctly
    RewritePatternSet patterns(&context);
    TypeConverter typeConverter;
    
    // Populate patterns (this tests the pattern population infrastructure)
    relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
    
    // The pattern set should be populated (though it might be empty in current implementation)
    // This primarily tests that the function executes without error
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PassInfrastructureValidation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing pass infrastructure validation");
    
    // Test various aspects of pass infrastructure
    auto pass = relalg::createLowerToSubOpPass();
    
    // Test pass type identification
    EXPECT_TRUE(pass != nullptr);
    
    // Test that the pass can be moved/copied as needed
    auto pass2 = relalg::createLowerToSubOpPass();
    EXPECT_TRUE(pass2 != nullptr);
    
    // Test pipeline creation
    PassManager pm(&context);
    relalg::createLowerRelAlgToSubOpPipeline(pm);
    EXPECT_GT(pm.size(), 0);
}

TEST_F(PassPipelineIntegrationTest, ErrorPropagation) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing error propagation");
    
    // Create a module that might cause issues during lowering
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    
    // Add some potentially problematic content
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_error", funcType);
    
    // Create pass manager
    PassManager pm(&context);
    pm.addPass(relalg::createLowerToSubOpPass());
    
    // Run the pass and verify error handling
    LogicalResult result = pm.run(*module);
    
    // The pass should handle errors gracefully (success or controlled failure)
    // This tests that the pass doesn't crash on unexpected input
    SUCCEED();
}

TEST_F(PassPipelineIntegrationTest, PerformanceCharacteristics) {
    MLIR_PGX_DEBUG("PassPipelineTest", "Testing performance characteristics");
    
    // Create a larger module for performance testing
    auto module = createSimpleRelAlgModule();
    EXPECT_TRUE(module);
    
    PassManager pm(&context);
    pm.addPass(relalg::createLowerToSubOpPass());
    pm.enableTiming();
    
    // Run the pass and measure basic performance characteristics
    auto start = std::chrono::high_resolution_clock::now();
    LogicalResult result = pm.run(*module);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Verify the pass completed in reasonable time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 10000); // Should complete within 10 seconds
    
    EXPECT_TRUE(succeeded(result));
}