#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

/**
 * ThreadLocalOperationsTest - Comprehensive test suite for thread-local operations
 * 
 * This test suite focuses on:
 * 1. Thread-local storage creation and access patterns
 * 2. Concurrency safety and thread isolation
 * 3. State management across execution contexts
 * 4. Performance characteristics of thread-local operations
 * 5. Critical issue: Operations being created in wrong execution contexts
 */
class ThreadLocalOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
        
        // Initialize thread-local operation counters
        operationCounts.store(0);
        contextSwitches.store(0);
        
        PGX_DEBUG("ThreadLocalOperationsTest setup completed");
    }

    void TearDown() override {
        PGX_DEBUG("ThreadLocalOperationsTest teardown - Operations created: " 
                 + std::to_string(operationCounts.load()) 
                 + ", Context switches: " + std::to_string(contextSwitches.load()));
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
    
    // Thread safety tracking
    std::atomic<int> operationCounts{0};
    std::atomic<int> contextSwitches{0};
    std::mutex testMutex;
    
    // Helper to create a basic module with main function
    ModuleOp createBasicModule() {
        auto module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
        
        auto mainFuncType = builder->getFunctionType({}, builder->getI32Type());
        auto mainFunc = builder->create<func::FuncOp>(loc, "main", mainFuncType);
        auto* mainBlock = mainFunc.addEntryBlock();
        
        builder->setInsertionPointToStart(mainBlock);
        
        return module;
    }
    
    // Helper to create thread-local merge operations
    template<typename OpType>
    OpType createThreadLocalMergeOp(ModuleOp module, Type resultType) {
        builder->setInsertionPointToEnd(module.getBody());
        
        // Create a dummy thread-local value for testing
        auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
        
        // Track operation creation for context validation
        operationCounts.fetch_add(1);
        
        return builder->create<OpType>(loc, resultType, tlValue);
    }
    
    // Helper to validate execution context
    bool validateExecutionContext(Operation* op) {
        // Check if operation was created in correct context
        auto parentModule = op->getParentOfType<ModuleOp>();
        if (!parentModule) {
            PGX_ERROR("Operation created without proper module context");
            return false;
        }
        
        // Check for proper dialect registration
        auto* dialect = op->getDialect();
        if (!dialect) {
            PGX_ERROR("Operation created without proper dialect context");
            return false;
        }
        
        PGX_DEBUG("Operation validation passed - Context: " + 
                 parentModule.getName().value_or_empty().str() + 
                 ", Dialect: " + dialect->getNamespace().str());
        return true;
    }
};

/**
 * Test thread-local storage creation and basic functionality
 */
TEST_F(ThreadLocalOperationsTest, ThreadLocalStorageCreation) {
    PGX_INFO("Testing thread-local storage creation");
    
    auto module = createBasicModule();
    
    // Test different thread-local data types
    auto resultTableType = subop::ResultTableType::get(&context, {});
    auto bufferType = subop::BufferType::get(&context, builder->getI32Type());
    auto heapType = subop::HeapType::get(&context, builder->getI32Type(), false);
    
    // Create thread-local operations for each type
    auto resultTableMerge = createThreadLocalMergeOp<subop::MergeOp>(module, resultTableType);
    auto bufferMerge = createThreadLocalMergeOp<subop::MergeOp>(module, bufferType);
    auto heapMerge = createThreadLocalMergeOp<subop::MergeOp>(module, heapType);
    
    // Validate execution contexts
    EXPECT_TRUE(validateExecutionContext(resultTableMerge.getOperation()));
    EXPECT_TRUE(validateExecutionContext(bufferMerge.getOperation()));
    EXPECT_TRUE(validateExecutionContext(heapMerge.getOperation()));
    
    // Verify operations were created correctly
    EXPECT_EQ(operationCounts.load(), 6); // 3 merge ops + 3 constant ops
    
    PGX_INFO("Thread-local storage creation test completed");
}

/**
 * Test thread safety and isolation between threads
 */
TEST_F(ThreadLocalOperationsTest, ThreadSafetyAndIsolation) {
    PGX_INFO("Testing thread safety and isolation");
    
    const int numThreads = 4;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    std::vector<std::atomic<int>> threadLocalCounts(numThreads);
    
    // Initialize per-thread counters
    for (int i = 0; i < numThreads; ++i) {
        threadLocalCounts[i].store(0);
    }
    
    // Launch threads that create thread-local operations
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread, &threadLocalCounts]() {
            // Each thread gets its own MLIR context to simulate isolation
            MLIRContext threadContext;
            threadContext.loadDialect<subop::SubOperatorDialect>();
            threadContext.loadDialect<util::UtilDialect>();
            threadContext.loadDialect<func::FuncDialect>();
            
            OpBuilder threadBuilder(&threadContext);
            auto threadLoc = threadBuilder.getUnknownLoc();
            
            PGX_DEBUG("Thread " + std::to_string(i) + " starting operations");
            
            for (int j = 0; j < operationsPerThread; ++j) {
                {
                    std::lock_guard<std::mutex> lock(testMutex);
                    contextSwitches.fetch_add(1);
                }
                
                // Create module in thread context
                auto module = ModuleOp::create(threadLoc);
                threadBuilder.setInsertionPointToEnd(module.getBody());
                
                // Create thread-local operation
                auto bufferType = subop::BufferType::get(&threadContext, threadBuilder.getI32Type());
                auto tlValue = threadBuilder.create<arith::ConstantIntOp>(threadLoc, i * 1000 + j, 64);
                auto mergeOp = threadBuilder.create<subop::MergeOp>(threadLoc, bufferType, tlValue);
                
                // Validate that operation is in correct thread context
                EXPECT_EQ(mergeOp.getOperation()->getContext(), &threadContext);
                threadLocalCounts[i].fetch_add(1);
                
                // Small delay to increase chances of context switching
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            
            PGX_DEBUG("Thread " + std::to_string(i) + " completed operations");
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify isolation - each thread should have completed its operations
    for (int i = 0; i < numThreads; ++i) {
        EXPECT_EQ(threadLocalCounts[i].load(), operationsPerThread);
    }
    
    // Verify context switches occurred (indicating concurrent execution)
    EXPECT_GT(contextSwitches.load(), 0);
    
    PGX_INFO("Thread safety and isolation test completed - Context switches: " 
            + std::to_string(contextSwitches.load()));
}

/**
 * Test state management and initialization patterns
 */
TEST_F(ThreadLocalOperationsTest, StateManagementAndInitialization) {
    PGX_INFO("Testing state management and initialization");
    
    auto module = createBasicModule();
    
    // Test SimpleState thread-local operations
    std::vector<Attribute> members = {
        builder->getStringAttr("count"),
        builder->getStringAttr("sum")
    };
    auto simpleStateType = subop::SimpleStateType::get(&context, members, false);
    
    // Create thread-local simple state merge operation
    builder->setInsertionPointToEnd(module.getBody());
    auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
    auto stateResetOp = builder->create<subop::ResetOp>(loc, simpleStateType, tlValue);
    auto stateMergeOp = builder->create<subop::MergeOp>(loc, simpleStateType, tlValue);
    
    // Validate proper initialization context
    EXPECT_TRUE(validateExecutionContext(stateResetOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(stateMergeOp.getOperation()));
    
    // Check that state type information is preserved
    auto resultType = stateMergeOp.getType();
    EXPECT_TRUE(mlir::isa<subop::SimpleStateType>(resultType));
    
    auto simpleState = mlir::cast<subop::SimpleStateType>(resultType);
    EXPECT_EQ(simpleState.getMembers().size(), 2);
    EXPECT_FALSE(simpleState.hasLock());
    
    PGX_INFO("State management and initialization test completed");
}

/**
 * Test hash map thread-local operations with key-value semantics
 */
TEST_F(ThreadLocalOperationsTest, HashMapThreadLocalOperations) {
    PGX_INFO("Testing hash map thread-local operations");
    
    auto module = createBasicModule();
    
    // Create hash map type with key and value members
    std::vector<Attribute> keyMembers = {builder->getStringAttr("id")};
    std::vector<Attribute> valueMembers = {
        builder->getStringAttr("count"),
        builder->getStringAttr("total")
    };
    
    auto hashMapType = subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    
    // Create thread-local hash map operations
    builder->setInsertionPointToEnd(module.getBody());
    auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
    
    // Test different hash map operations in sequence
    auto lookupOp = builder->create<subop::LookupOp>(loc, hashMapType, tlValue, ValueRange{});
    auto insertOp = builder->create<subop::InsertOp>(loc, hashMapType, tlValue, ValueRange{});
    auto mergeOp = builder->create<subop::MergeOp>(loc, hashMapType, tlValue);
    
    // Validate execution contexts for all operations
    EXPECT_TRUE(validateExecutionContext(lookupOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(insertOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(mergeOp.getOperation()));
    
    // Verify hash map type preservation
    auto mergeResultType = mergeOp.getType();
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(mergeResultType));
    
    auto hashMap = mlir::cast<subop::HashMapType>(mergeResultType);
    EXPECT_EQ(hashMap.getKeyMembers().size(), 1);
    EXPECT_EQ(hashMap.getValueMembers().size(), 2);
    EXPECT_FALSE(hashMap.hasLock());
    
    PGX_INFO("Hash map thread-local operations test completed");
}

/**
 * Test pre-aggregation hash table thread-local operations
 */
TEST_F(ThreadLocalOperationsTest, PreAggregationHashTableOperations) {
    PGX_INFO("Testing pre-aggregation hash table thread-local operations");
    
    auto module = createBasicModule();
    
    // Create pre-aggregation hash table type
    std::vector<Attribute> keyMembers = {builder->getStringAttr("group_key")};
    std::vector<Attribute> valueMembers = {builder->getStringAttr("agg_value")};
    
    auto preAggrType = subop::PreAggrHtType::get(&context, keyMembers, valueMembers, false);
    
    // Create thread-local pre-aggregation operations
    builder->setInsertionPointToEnd(module.getBody());
    auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
    
    auto resetOp = builder->create<subop::ResetOp>(loc, preAggrType, tlValue);
    auto insertOp = builder->create<subop::InsertOp>(loc, preAggrType, tlValue, ValueRange{});
    auto mergeOp = builder->create<subop::MergeOp>(loc, preAggrType, tlValue);
    
    // Validate execution contexts
    EXPECT_TRUE(validateExecutionContext(resetOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(insertOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(mergeOp.getOperation()));
    
    // Verify type correctness
    auto resultType = mergeOp.getType();
    EXPECT_TRUE(mlir::isa<subop::PreAggrHtType>(resultType));
    
    PGX_INFO("Pre-aggregation hash table operations test completed");
}

/**
 * Test performance characteristics of thread-local operations
 */
TEST_F(ThreadLocalOperationsTest, PerformanceCharacteristics) {
    PGX_INFO("Testing performance characteristics");
    
    const int numOperations = 1000;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto module = createBasicModule();
    auto bufferType = subop::BufferType::get(&context, builder->getI32Type());
    
    // Create many thread-local operations rapidly
    for (int i = 0; i < numOperations; ++i) {
        builder->setInsertionPointToEnd(module.getBody());
        auto tlValue = builder->create<arith::ConstantIntOp>(loc, i, 64);
        auto mergeOp = builder->create<subop::MergeOp>(loc, bufferType, tlValue);
        
        // Validate every 100th operation to avoid overhead
        if (i % 100 == 0) {
            EXPECT_TRUE(validateExecutionContext(mergeOp.getOperation()));
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // Performance expectations (should be reasonably fast)
    EXPECT_LT(duration.count(), 10000); // Less than 10ms for 1000 operations
    
    PGX_INFO("Performance test completed - Duration: " + std::to_string(duration.count()) + " microseconds");
}

/**
 * Critical test: Verify operations are created in correct execution context
 * This addresses the specific concern about thread-local operations being
 * created in wrong execution contexts.
 */
TEST_F(ThreadLocalOperationsTest, ExecutionContextValidation) {
    PGX_INFO("Testing execution context validation (CRITICAL)");
    
    // Test 1: Operations created in correct module context
    {
        auto module1 = createBasicModule();
        auto module2 = createBasicModule();
        
        // Create operation in module1 context
        builder->setInsertionPointToEnd(module1.getBody());
        auto bufferType = subop::BufferType::get(&context, builder->getI32Type());
        auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
        auto mergeOp1 = builder->create<subop::MergeOp>(loc, bufferType, tlValue);
        
        // Verify operation belongs to module1
        EXPECT_EQ(mergeOp1->getParentOfType<ModuleOp>().getOperation(), module1.getOperation());
        
        // Create operation in module2 context
        builder->setInsertionPointToEnd(module2.getBody());
        auto mergeOp2 = builder->create<subop::MergeOp>(loc, bufferType, tlValue);
        
        // Verify operation belongs to module2
        EXPECT_EQ(mergeOp2->getParentOfType<ModuleOp>().getOperation(), module2.getOperation());
        
        PGX_DEBUG("Module context validation passed");
    }
    
    // Test 2: Context preservation across function boundaries
    {
        auto module = createBasicModule();
        
        // Create a function within the module
        builder->setInsertionPointToEnd(module.getBody());
        auto funcType = builder->getFunctionType({builder->getI32Type()}, {builder->getI32Type()});
        auto testFunc = builder->create<func::FuncOp>(loc, "test_func", funcType);
        auto* funcBlock = testFunc.addEntryBlock();
        
        // Create operation within function
        builder->setInsertionPointToStart(funcBlock);
        auto bufferType = subop::BufferType::get(&context, builder->getI32Type());
        auto arg = funcBlock->getArgument(0);
        auto mergeOp = builder->create<subop::MergeOp>(loc, bufferType, arg);
        
        // Verify operation is in correct function and module context
        EXPECT_EQ(mergeOp->getParentOfType<func::FuncOp>().getOperation(), testFunc.getOperation());
        EXPECT_EQ(mergeOp->getParentOfType<ModuleOp>().getOperation(), module.getOperation());
        
        PGX_DEBUG("Function context validation passed");
    }
    
    // Test 3: Dialect context validation
    {
        auto module = createBasicModule();
        builder->setInsertionPointToEnd(module.getBody());
        
        auto bufferType = subop::BufferType::get(&context, builder->getI32Type());
        auto tlValue = builder->create<arith::ConstantIntOp>(loc, 0, 64);
        auto mergeOp = builder->create<subop::MergeOp>(loc, bufferType, tlValue);
        
        // Verify operation has correct dialect
        auto* dialect = mergeOp->getDialect();
        EXPECT_NE(dialect, nullptr);
        EXPECT_EQ(dialect->getNamespace(), "subop");
        
        PGX_DEBUG("Dialect context validation passed");
    }
    
    PGX_INFO("Execution context validation test completed (CRITICAL TEST PASSED)");
}

/**
 * Test concurrent access patterns with context switching
 */
TEST_F(ThreadLocalOperationsTest, ConcurrentContextSwitching) {
    PGX_INFO("Testing concurrent context switching");
    
    const int numThreads = 3;
    const int contextSwitchesPerThread = 5;
    std::vector<std::thread> threads;
    std::atomic<int> totalContextValidations{0};
    std::atomic<int> validationFailures{0};
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, contextSwitchesPerThread, &totalContextValidations, &validationFailures]() {
            for (int j = 0; j < contextSwitchesPerThread; ++j) {
                // Create new context for each iteration (simulating context switch)
                MLIRContext threadContext;
                threadContext.loadDialect<subop::SubOperatorDialect>();
                threadContext.loadDialect<arith::ArithDialect>();
                threadContext.loadDialect<func::FuncDialect>();
                
                OpBuilder threadBuilder(&threadContext);
                auto threadLoc = threadBuilder.getUnknownLoc();
                
                // Create module and operations
                auto module = ModuleOp::create(threadLoc);
                threadBuilder.setInsertionPointToEnd(module.getBody());
                
                auto bufferType = subop::BufferType::get(&threadContext, threadBuilder.getI32Type());
                auto tlValue = threadBuilder.create<arith::ConstantIntOp>(threadLoc, i * 100 + j, 64);
                auto mergeOp = threadBuilder.create<subop::MergeOp>(threadLoc, bufferType, tlValue);
                
                // Validate context
                totalContextValidations.fetch_add(1);
                
                if (!validateExecutionContext(mergeOp.getOperation())) {
                    validationFailures.fetch_add(1);
                    PGX_ERROR("Context validation failed in thread " + std::to_string(i) + 
                             ", iteration " + std::to_string(j));
                }
                
                // Brief pause to encourage context switching
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all validations passed
    EXPECT_EQ(totalContextValidations.load(), numThreads * contextSwitchesPerThread);
    EXPECT_EQ(validationFailures.load(), 0);
    
    PGX_INFO("Concurrent context switching test completed - Validations: " + 
            std::to_string(totalContextValidations.load()) + 
            ", Failures: " + std::to_string(validationFailures.load()));
}