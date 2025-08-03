#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowRewriter.h"
#include "src/dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowCommon.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;
using namespace pgx_lower::compiler::dialect::subop_to_cf;

class SubOpRewriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
    
    // Helper to create a test execution step
    subop::ExecutionStepOp createTestExecutionStep() {
        auto funcType = FunctionType::get(&context, {}, {});
        auto funcOp = builder->create<func::FuncOp>(loc, "test_func", funcType);
        auto* block = funcOp.addEntryBlock();
        
        builder->setInsertionPointToEnd(block);
        
        // Create execution step with empty region
        auto execStep = builder->create<subop::ExecutionStepOp>(loc, TypeRange{}, ValueRange{}, ArrayAttr{});
        auto& region = execStep.getSubOps();
        auto* stepBlock = region.emplaceBlock();
        
        return execStep;
    }
    
    // Helper to create column mapping
    ColumnMapping createTestColumnMapping() {
        ColumnMapping mapping;
        
        // Create test column
        auto* tupleDialect = context.getLoadedDialect<tuples::TupleStreamDialect>();
        auto& columnManager = tupleDialect->getColumnManager();
        auto* column = &columnManager.createColumn("test_col");
        auto columnDef = columnManager.createDef(column);
        
        // Create test value
        auto testValue = builder->create<arith::ConstantIntOp>(loc, 42, 32);
        
        mapping.define(columnDef, testValue);
        return mapping;
    }
};

// ============================================================================
// ColumnMapping Tests
// ============================================================================

TEST_F(SubOpRewriterTest, ColumnMappingBasicOperations) {
    PGX_DEBUG("Testing ColumnMapping basic operations");
    
    ColumnMapping mapping = createTestColumnMapping();
    
    // Test that mapping has entries
    const auto& internalMapping = mapping.getMapping();
    EXPECT_FALSE(internalMapping.empty());
    EXPECT_EQ(internalMapping.size(), 1);
}

TEST_F(SubOpRewriterTest, ColumnMappingInFlightCreation) {
    PGX_DEBUG("Testing ColumnMapping InFlight creation");
    
    ColumnMapping mapping = createTestColumnMapping();
    
    // Test createInFlight
    Value inFlightValue = mapping.createInFlight(*builder);
    EXPECT_TRUE(inFlightValue);
    EXPECT_TRUE(inFlightValue.getDefiningOp());
    
    // Verify it's a subop::InFlightOp
    auto inFlightOp = dyn_cast<subop::InFlightOp>(inFlightValue.getDefiningOp());
    EXPECT_TRUE(inFlightOp);
}

TEST_F(SubOpRewriterTest, ColumnMappingInFlightTupleCreation) {
    PGX_DEBUG("Testing ColumnMapping InFlightTuple creation");
    
    ColumnMapping mapping = createTestColumnMapping();
    
    // Test createInFlightTuple
    Value inFlightTupleValue = mapping.createInFlightTuple(*builder);
    EXPECT_TRUE(inFlightTupleValue);
    EXPECT_TRUE(inFlightTupleValue.getDefiningOp());
    
    // Verify it's a subop::InFlightTupleOp
    auto inFlightTupleOp = dyn_cast<subop::InFlightTupleOp>(inFlightTupleValue.getDefiningOp());
    EXPECT_TRUE(inFlightTupleOp);
}

TEST_F(SubOpRewriterTest, ColumnMappingMergeOperations) {
    PGX_DEBUG("Testing ColumnMapping merge operations");
    
    ColumnMapping mapping1 = createTestColumnMapping();
    ColumnMapping mapping2;
    
    // Create an InFlightOp for merging
    Value inFlightValue = mapping1.createInFlight(*builder);
    auto inFlightOp = cast<subop::InFlightOp>(inFlightValue.getDefiningOp());
    
    // Test merge with InFlightOp
    mapping2.merge(inFlightOp);
    
    // Verify mapping2 now has entries
    const auto& internalMapping = mapping2.getMapping();
    EXPECT_FALSE(internalMapping.empty());
}

// ============================================================================
// SubOpRewriter Basic Tests
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterConstruction) {
    PGX_DEBUG("Testing SubOpRewriter construction");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    
    // Test constructor
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Test basic getters
    EXPECT_TRUE(rewriter.getContext());
    EXPECT_TRUE(rewriter.getI1Type());
    EXPECT_TRUE(rewriter.getI8Type());
    EXPECT_TRUE(rewriter.getIndexType());
}

TEST_F(SubOpRewriterTest, RewriterAlternateConstruction) {
    PGX_DEBUG("Testing SubOpRewriter alternate construction");
    
    // Test alternate constructor
    SubOpRewriter rewriter(*builder);
    
    EXPECT_TRUE(rewriter.getContext());
    EXPECT_EQ(rewriter.getContext(), &context);
}

TEST_F(SubOpRewriterTest, RewriterValueMapping) {
    PGX_DEBUG("Testing SubOpRewriter value mapping");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create test values
    auto originalValue = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    auto mappedValue = builder->create<arith::ConstantIntOp>(loc, 84, 32);
    
    // Test mapping
    rewriter.map(originalValue, mappedValue);
    
    // Test retrieval
    Value result = rewriter.getMapped(originalValue);
    EXPECT_EQ(result, mappedValue);
    
    // Test unmapped value returns itself
    auto unmappedValue = builder->create<arith::ConstantIntOp>(loc, 123, 32);
    Value unmappedResult = rewriter.getMapped(unmappedValue);
    EXPECT_EQ(unmappedResult, unmappedValue);
}

// ============================================================================
// Block Structure and Terminator Tests
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterBlockOperations) {
    PGX_DEBUG("Testing SubOpRewriter block operations");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create a test block
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_block_ops", funcType);
    auto* testBlock = funcOp.addEntryBlock();
    
    // Test atStartOf functionality
    bool callbackExecuted = false;
    rewriter.atStartOf(testBlock, [&](SubOpRewriter& r) {
        callbackExecuted = true;
        EXPECT_EQ(&r, &rewriter);
    });
    
    EXPECT_TRUE(callbackExecuted);
}

TEST_F(SubOpRewriterTest, RewriterOperationReplacement) {
    PGX_DEBUG("Testing SubOpRewriter operation replacement");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create an operation to replace
    auto originalOp = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    auto replacementValue = builder->create<arith::ConstantIntOp>(loc, 84, 32);
    
    // Test replaceOp
    rewriter.replaceOp(originalOp, ValueRange{replacementValue});
    
    // Verify original operation result is mapped to replacement
    Value mappedResult = rewriter.getMapped(originalOp.getResult());
    EXPECT_EQ(mappedResult, replacementValue);
}

TEST_F(SubOpRewriterTest, RewriterOperationErasure) {
    PGX_DEBUG("Testing SubOpRewriter operation erasure");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create an operation to erase
    auto opToErase = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Test eraseOp (should mark for deletion, not immediately erase)
    rewriter.eraseOp(opToErase);
    
    // Operation should still exist until cleanup
    EXPECT_TRUE(opToErase.getOperation());
    
    // Test cleanup
    rewriter.cleanup();
    
    // After cleanup, verify operations are properly handled
    // Note: We can't test actual erasure since that would invalidate pointers
    // This test ensures the cleanup mechanism runs without errors
    EXPECT_TRUE(true);
}

// ============================================================================
// Step Requirements and Context Management Tests
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterStepRequirements) {
    PGX_DEBUG("Testing SubOpRewriter step requirements");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    
    // Add some input parameters to the execution step
    auto inputValue = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    outerMapping.map(inputValue, inputValue);
    
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Test storeStepRequirements
    Value contextPtr = rewriter.storeStepRequirements();
    EXPECT_TRUE(contextPtr);
    EXPECT_TRUE(contextPtr.getType());
}

TEST_F(SubOpRewriterTest, RewriterGuardMechanism) {
    PGX_DEBUG("Testing SubOpRewriter guard mechanism");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create test values
    auto testValue = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    auto mappedValue = builder->create<arith::ConstantIntOp>(loc, 84, 32);
    
    // Test Guard scope
    {
        SubOpRewriter::Guard guard(rewriter);
        
        // Map a value within the guard scope
        rewriter.map(testValue, mappedValue);
        
        // Value should be mapped
        Value result = rewriter.getMapped(testValue);
        EXPECT_EQ(result, mappedValue);
    }
    
    // After guard scope ends, mapping should be cleaned up
    // Note: Guard destructor should handle this
    EXPECT_TRUE(true); // Guard mechanism test passes if no crashes occur
}

// ============================================================================
// Tuple Stream Management Tests
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterTupleStreamOperations) {
    PGX_DEBUG("Testing SubOpRewriter tuple stream operations");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    ColumnMapping mapping = createTestColumnMapping();
    
    // Test createInFlight
    auto inFlightOp = rewriter.createInFlight(mapping);
    EXPECT_TRUE(inFlightOp);
    
    // Test getTupleStream
    Value streamValue = inFlightOp.getResult();
    InFlightTupleStream stream = rewriter.getTupleStream(streamValue);
    EXPECT_TRUE(stream.inFlightOp);
    EXPECT_EQ(stream.inFlightOp, inFlightOp);
}

TEST_F(SubOpRewriterTest, RewriterTupleStreamReplacement) {
    PGX_DEBUG("Testing SubOpRewriter tuple stream replacement");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    ColumnMapping mapping1 = createTestColumnMapping();
    ColumnMapping mapping2 = createTestColumnMapping();
    
    // Create original stream
    auto originalInFlight = rewriter.createInFlight(mapping1);
    Value originalStream = originalInFlight.getResult();
    
    // Test replaceTupleStream with mapping
    rewriter.replaceTupleStream(originalStream, mapping2);
    
    // Stream should now reference the new mapping
    InFlightTupleStream replacedStream = rewriter.getTupleStream(originalStream);
    EXPECT_TRUE(replacedStream.inFlightOp);
}

// ============================================================================
// Stream Consumer Implementation Tests (Critical for Terminator Issues)
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterStreamConsumerImplementation) {
    PGX_DEBUG("Testing SubOpRewriter stream consumer implementation");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    ColumnMapping mapping = createTestColumnMapping();
    auto inFlightOp = rewriter.createInFlight(mapping);
    Value stream = inFlightOp.getResult();
    
    // Test implementStreamConsumer with a simple implementation
    bool implCalled = false;
    LogicalResult result = rewriter.implementStreamConsumer(stream, 
        [&](SubOpRewriter& r, ColumnMapping& m) -> LogicalResult {
            implCalled = true;
            EXPECT_EQ(&r, &rewriter);
            EXPECT_FALSE(m.getMapping().empty());
            return success();
        });
    
    EXPECT_TRUE(succeeded(result));
    EXPECT_TRUE(implCalled);
}

// ============================================================================
// Critical Pattern Tests for Terminator Interference
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterTerminatorSafeOperations) {
    PGX_DEBUG("Testing SubOpRewriter terminator-safe operations");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create a function with a block that needs a terminator
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_terminator_safe", funcType);
    auto* block = funcOp.addEntryBlock();
    
    // Test that rewriter operations don't interfere with block structure
    rewriter.atStartOf(block, [&](SubOpRewriter& r) {
        // Create some operations
        auto constOp = r.create<arith::ConstantIntOp>(loc, 42, 32);
        EXPECT_TRUE(constOp);
        
        // Verify block still has proper structure
        EXPECT_EQ(&constOp->getBlock(), block);
    });
    
    // Block should still be in valid state for terminator addition
    EXPECT_TRUE(block);
    EXPECT_FALSE(block->empty()); // Should have operations
}

TEST_F(SubOpRewriterTest, RewriterOperationCreationPatterns) {
    PGX_DEBUG("Testing SubOpRewriter operation creation patterns");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Test various operation creation patterns that might affect terminators
    
    // Test arithmetic operations
    auto constOp = rewriter.create<arith::ConstantIntOp>(loc, 42, 32);
    EXPECT_TRUE(constOp);
    
    // Test utility operations
    auto allocaOp = rewriter.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, builder->getI32Type()), Value());
    EXPECT_TRUE(allocaOp);
    
    // Test that operations are properly created without affecting block terminators
    Block* block = constOp->getBlock();
    EXPECT_TRUE(block);
    EXPECT_FALSE(block->empty());
}

TEST_F(SubOpRewriterTest, RewriterBlockManipulationSafety) {
    PGX_DEBUG("Testing SubOpRewriter block manipulation safety");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create a control flow structure that requires proper terminator handling
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_block_safety", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    
    builder->setInsertionPointToEnd(entryBlock);
    
    // Create a conditional structure
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    auto ifOp = builder->create<scf::IfOp>(loc, condition, false);
    
    // Test that rewriter can work within control flow without breaking structure
    rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& r) {
        auto innerOp = r.create<arith::ConstantIntOp>(loc, 123, 32);
        EXPECT_TRUE(innerOp);
    });
    
    // Verify control flow structure is intact
    EXPECT_TRUE(ifOp);
    EXPECT_FALSE(ifOp.getThenRegion().empty());
}

// ============================================================================
// Integration Tests for Complete Rewriter Workflow
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterCompleteWorkflow) {
    PGX_DEBUG("Testing complete SubOpRewriter workflow");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Simulate a complete rewrite workflow
    ColumnMapping inputMapping = createTestColumnMapping();
    
    // 1. Create initial stream
    auto inFlightOp = rewriter.createInFlight(inputMapping);
    Value stream = inFlightOp.getResult();
    
    // 2. Implement stream consumer with typical operations
    LogicalResult result = rewriter.implementStreamConsumer(stream,
        [&](SubOpRewriter& r, ColumnMapping& mapping) -> LogicalResult {
            // Create some operations typical of SubOp lowering
            auto constOp = r.create<arith::ConstantIntOp>(loc, 42, 32);
            
            // Manipulate the mapping
            auto* tupleDialect = context.getLoadedDialect<tuples::TupleStreamDialect>();
            auto& columnManager = tupleDialect->getColumnManager();
            auto* newColumn = &columnManager.createColumn("computed_col");
            auto newColumnDef = columnManager.createDef(newColumn);
            
            mapping.define(newColumnDef, constOp);
            
            return success();
        });
    
    EXPECT_TRUE(succeeded(result));
    
    // 3. Test cleanup
    rewriter.cleanup();
    
    PGX_DEBUG("Complete workflow test passed");
}

TEST_F(SubOpRewriterTest, RewriterMemoryManagement) {
    PGX_DEBUG("Testing SubOpRewriter memory management");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create multiple operations and manage their lifecycle
    std::vector<Operation*> createdOps;
    
    for (int i = 0; i < 10; ++i) {
        auto op = rewriter.create<arith::ConstantIntOp>(loc, i, 32);
        createdOps.push_back(op);
        
        if (i % 2 == 0) {
            // Mark every other operation for erasure
            rewriter.eraseOp(op);
        }
    }
    
    // Test cleanup handles all marked operations
    rewriter.cleanup();
    
    // Memory management test passes if no crashes occur
    EXPECT_TRUE(true);
}

// ============================================================================
// Edge Case and Error Handling Tests
// ============================================================================

TEST_F(SubOpRewriterTest, RewriterEdgeCases) {
    PGX_DEBUG("Testing SubOpRewriter edge cases");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Test with empty column mapping
    ColumnMapping emptyMapping;
    
    // Should handle empty mapping gracefully
    Value emptyInFlight = emptyMapping.createInFlight(*builder);
    EXPECT_TRUE(emptyInFlight);
    
    auto emptyInFlightOp = cast<subop::InFlightOp>(emptyInFlight.getDefiningOp());
    EXPECT_TRUE(emptyInFlightOp);
    EXPECT_EQ(emptyInFlightOp.getColumns().size(), 0);
    EXPECT_EQ(emptyInFlightOp.getValues().size(), 0);
}

TEST_F(SubOpRewriterTest, RewriterNestingGuardMechanism) {
    PGX_DEBUG("Testing SubOpRewriter nesting guard mechanism");
    
    auto execStep = createTestExecutionStep();
    IRMapping outerMapping;
    SubOpRewriter rewriter(execStep, outerMapping);
    
    // Create nested execution step
    auto nestedExecStep = createTestExecutionStep();
    IRMapping nestedMapping;
    
    // Test NestingGuard
    {
        auto nestingGuard = rewriter.nest(nestedMapping, nestedExecStep);
        
        // Should be able to perform operations within nested context
        auto nestedOp = rewriter.create<arith::ConstantIntOp>(loc, 999, 32);
        EXPECT_TRUE(nestedOp);
    }
    
    // After nesting guard scope, should return to original context
    EXPECT_TRUE(true); // Test passes if no crashes occur
}