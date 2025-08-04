// Unit tests for thread-local operations
// Tests basic thread-local operation creation and compilation

#include <gtest/gtest.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

/**
 * ThreadLocalOperationsTest - Test suite for thread-local operations
 * 
 * Tests basic functionality of thread-local operations in the SubOp dialect
 */
class ThreadLocalOperationsTest : public ::testing::Test {
protected:
    // Add default constructor for GoogleTest compatibility
    ThreadLocalOperationsTest() = default;
    
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
        
        PGX_DEBUG("ThreadLocalOperationsTest setup completed");
    }

    void TearDown() override {
        PGX_DEBUG("ThreadLocalOperationsTest teardown completed");
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc = UnknownLoc::get(&context);
    
    // Helper to create a basic module
    ModuleOp createBasicModule() {
        auto module = ModuleOp::create(loc);
        return module;
    }
    
    // Helper to create state members attribute
    subop::StateMembersAttr createStateMembers(ArrayRef<StringRef> names, ArrayRef<Type> types) {
        SmallVector<Attribute> nameAttrs;
        SmallVector<Attribute> typeAttrs;
        
        for (auto name : names) {
            nameAttrs.push_back(builder->getStringAttr(name));
        }
        for (auto type : types) {
            typeAttrs.push_back(TypeAttr::get(type));
        }
        
        auto nameArray = builder->getArrayAttr(nameAttrs);
        auto typeArray = builder->getArrayAttr(typeAttrs);
        
        return subop::StateMembersAttr::get(&context, nameArray, typeArray);
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
        
        PGX_DEBUG("Operation validation passed");
        return true;
    }
};

/**
 * Test thread-local storage creation and basic functionality
 */
TEST_F(ThreadLocalOperationsTest, ThreadLocalStorageCreation) {
    PGX_INFO("Testing thread-local storage creation");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create StateMembers for simple state
    auto i64Type = builder->getI64Type();
    auto simpleStateMembers = createStateMembers({"count", "sum"}, {i64Type, i64Type});
    
    // Create simple state type
    auto simpleStateType = subop::SimpleStateType::get(&context, simpleStateMembers);
    
    // Create thread-local type
    auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
    
    // Create thread-local operation
    auto createThreadLocalOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Validate execution contexts
    EXPECT_TRUE(validateExecutionContext(createThreadLocalOp.getOperation()));
    
    // Verify thread-local types are correct
    EXPECT_TRUE(mlir::isa<subop::ThreadLocalType>(createThreadLocalOp.getType()));
    
    PGX_INFO("Thread-local storage creation test completed");
    
    module.erase();
}

/**
 * Test state management and initialization patterns
 */
TEST_F(ThreadLocalOperationsTest, StateManagementAndInitialization) {
    PGX_INFO("Testing state management and initialization");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Test SimpleState thread-local operations
    auto simpleStateMembers = createStateMembers({"count", "sum"}, {builder->getI64Type(), builder->getI64Type()});
    auto simpleStateType = subop::SimpleStateType::get(&context, simpleStateMembers);
    
    // Create a simple state
    auto createSimpleStateOp = builder->create<subop::CreateSimpleStateOp>(loc, simpleStateType);
    
    // Create thread-local wrapper
    auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
    auto createThreadLocalOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Validate proper initialization context
    EXPECT_TRUE(validateExecutionContext(createSimpleStateOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(createThreadLocalOp.getOperation()));
    
    // Check that state type information is preserved
    auto resultStateType = createSimpleStateOp.getType();
    EXPECT_TRUE(mlir::isa<subop::SimpleStateType>(resultStateType));
    
    auto simpleState = mlir::cast<subop::SimpleStateType>(resultStateType);
    EXPECT_EQ(simpleState.getMembers().getNames().size(), 2);
    
    // Check thread-local type wrapping
    auto threadLocalResultType = createThreadLocalOp.getType();
    EXPECT_TRUE(mlir::isa<subop::ThreadLocalType>(threadLocalResultType));
    
    auto threadLocal = mlir::cast<subop::ThreadLocalType>(threadLocalResultType);
    EXPECT_TRUE(mlir::isa<subop::SimpleStateType>(threadLocal.getWrapped()));
    
    PGX_INFO("State management and initialization test completed");
    
    module.erase();
}

/**
 * Test hash map thread-local operations with key-value semantics
 */
TEST_F(ThreadLocalOperationsTest, HashMapThreadLocalOperations) {
    PGX_INFO("Testing hash map thread-local operations");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create hash map type with key and value members
    auto keyMembers = createStateMembers({"id"}, {builder->getI64Type()});
    auto valueMembers = createStateMembers({"count", "total"}, {builder->getI64Type(), builder->getI64Type()});
    
    auto hashMapType = subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    
    // Create the hash map
    auto createHashMapOp = builder->create<subop::GenericCreateOp>(loc, hashMapType);
    
    // Create thread-local wrapper
    auto threadLocalType = subop::ThreadLocalType::get(&context, hashMapType);
    auto createThreadLocalOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Validate execution contexts for all operations
    EXPECT_TRUE(validateExecutionContext(createHashMapOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(createThreadLocalOp.getOperation()));
    
    // Verify hash map type preservation
    auto hashMapResultType = createHashMapOp.getType();
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(hashMapResultType));
    
    auto hashMap = mlir::cast<subop::HashMapType>(hashMapResultType);
    EXPECT_EQ(hashMap.getKeyMembers().getNames().size(), 1);
    EXPECT_EQ(hashMap.getValueMembers().getNames().size(), 2);
    EXPECT_FALSE(hashMap.hasLock());
    
    // Verify thread-local wrapping
    auto threadLocalResultType = createThreadLocalOp.getType();
    EXPECT_TRUE(mlir::isa<subop::ThreadLocalType>(threadLocalResultType));
    
    auto threadLocal = mlir::cast<subop::ThreadLocalType>(threadLocalResultType);
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(threadLocal.getWrapped()));
    
    PGX_INFO("Hash map thread-local operations test completed");
    
    module.erase();
}

/**
 * Test pre-aggregation hash table thread-local operations
 */
TEST_F(ThreadLocalOperationsTest, PreAggregationHashTableOperations) {
    PGX_INFO("Testing pre-aggregation hash table thread-local operations");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create pre-aggregation hash table type
    auto keyMembers = createStateMembers({"group_key"}, {builder->getI64Type()});
    auto valueMembers = createStateMembers({"agg_value"}, {builder->getI64Type()});
    
    auto preAggrType = subop::PreAggrHtType::get(&context, keyMembers, valueMembers, false);
    
    // Create the pre-aggregation hash table
    auto createPreAggrOp = builder->create<subop::GenericCreateOp>(loc, preAggrType);
    
    // Create thread-local wrapper
    auto threadLocalType = subop::ThreadLocalType::get(&context, preAggrType);
    auto createThreadLocalOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Validate execution contexts
    EXPECT_TRUE(validateExecutionContext(createPreAggrOp.getOperation()));
    EXPECT_TRUE(validateExecutionContext(createThreadLocalOp.getOperation()));
    
    // Verify type correctness
    auto preAggrResultType = createPreAggrOp.getType();
    EXPECT_TRUE(mlir::isa<subop::PreAggrHtType>(preAggrResultType));
    
    auto preAggr = mlir::cast<subop::PreAggrHtType>(preAggrResultType);
    EXPECT_EQ(preAggr.getKeyMembers().getNames().size(), 1);
    EXPECT_EQ(preAggr.getValueMembers().getNames().size(), 1);
    EXPECT_FALSE(preAggr.hasLock());
    
    // Verify thread-local wrapping
    auto threadLocalResultType = createThreadLocalOp.getType();
    EXPECT_TRUE(mlir::isa<subop::ThreadLocalType>(threadLocalResultType));
    
    PGX_INFO("Pre-aggregation hash table operations test completed");
    
    module.erase();
}

/**
 * Critical test: Verify operations are created in correct execution context
 */
TEST_F(ThreadLocalOperationsTest, ExecutionContextValidation) {
    PGX_INFO("Testing execution context validation (CRITICAL)");
    
    // Test 1: Operations created in correct module context
    {
        auto module1 = createBasicModule();
        auto module2 = createBasicModule();
        
        // Create operation in module1 context
        builder->setInsertionPointToEnd(module1.getBody());
        auto bufferMembers = createStateMembers({"data"}, {builder->getI32Type()});
        auto simpleStateType = subop::SimpleStateType::get(&context, bufferMembers);
        auto threadLocalType1 = subop::ThreadLocalType::get(&context, simpleStateType);
        auto createOp1 = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType1);
        
        // Verify operation belongs to module1
        EXPECT_EQ(createOp1->getParentOfType<ModuleOp>().getOperation(), module1.getOperation());
        
        // Create operation in module2 context
        builder->setInsertionPointToEnd(module2.getBody());
        auto threadLocalType2 = subop::ThreadLocalType::get(&context, simpleStateType);
        auto createOp2 = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType2);
        
        // Verify operation belongs to module2
        EXPECT_EQ(createOp2->getParentOfType<ModuleOp>().getOperation(), module2.getOperation());
        
        PGX_DEBUG("Module context validation passed");
        
        module1.erase();
        module2.erase();
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
        auto bufferMembers = createStateMembers({"data"}, {builder->getI32Type()});
        auto simpleStateType = subop::SimpleStateType::get(&context, bufferMembers);
        auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
        auto createOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
        
        // Add function terminator
        auto arg = funcBlock->getArgument(0);
        builder->create<func::ReturnOp>(loc, arg);
        
        // Verify operation is in correct function and module context
        EXPECT_EQ(createOp->getParentOfType<func::FuncOp>().getOperation(), testFunc.getOperation());
        EXPECT_EQ(createOp->getParentOfType<ModuleOp>().getOperation(), module.getOperation());
        
        PGX_DEBUG("Function context validation passed");
        
        module.erase();
    }
    
    // Test 3: Dialect context validation
    {
        auto module = createBasicModule();
        builder->setInsertionPointToEnd(module.getBody());
        
        auto bufferMembers = createStateMembers({"data"}, {builder->getI32Type()});
        auto simpleStateType = subop::SimpleStateType::get(&context, bufferMembers);
        auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
        auto createOp = builder->create<subop::CreateThreadLocalOp>(loc, threadLocalType);
        
        // Verify operation has correct dialect
        auto* dialect = createOp->getDialect();
        EXPECT_NE(dialect, nullptr);
        EXPECT_EQ(dialect->getNamespace(), "subop");
        
        PGX_DEBUG("Dialect context validation passed");
        
        module.erase();
    }
    
    PGX_INFO("Execution context validation test completed (CRITICAL TEST PASSED)");
}

// Simple test for basic thread-local operation compilation
TEST(ThreadLocalOperationsBasicTest, BasicThreadLocalCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<tuples::TupleStreamDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create state members for simple state
    SmallVector<Attribute> nameAttrs = {builder.getStringAttr("count")};
    SmallVector<Attribute> typeAttrs = {TypeAttr::get(builder.getI64Type())};
    auto nameArray = builder.getArrayAttr(nameAttrs);
    auto typeArray = builder.getArrayAttr(typeAttrs);
    auto stateMembers = subop::StateMembersAttr::get(&context, nameArray, typeArray);
    
    // Create simple state type
    auto simpleStateType = subop::SimpleStateType::get(&context, stateMembers);
    
    // Create thread-local type
    auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
    
    // Create thread-local operation
    auto createThreadLocalOp = builder.create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Add terminator to the function
    builder.create<func::ReturnOp>(loc);
    
    // Verify the thread-local operation was created
    EXPECT_TRUE(createThreadLocalOp);
    EXPECT_TRUE(mlir::isa<subop::ThreadLocalType>(createThreadLocalOp.getType()));
    
    // Verify proper termination
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_TRUE(terminator->hasTrait<OpTrait::IsTerminator>());
    
    PGX_INFO("Basic thread-local operation test completed successfully");
    
    module.erase();
}

// Test for execution group creation with thread-local states
TEST(ThreadLocalOperationsBasicTest, ExecutionGroupWithThreadLocal) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<tuples::TupleStreamDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "execution_group_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create execution group
    auto executionGroupOp = builder.create<subop::ExecutionGroupOp>(loc, TypeRange{}, ValueRange{});
    
    // Create body for execution group
    auto& executionRegion = executionGroupOp.getSubOps();
    auto* executionBlock = builder.createBlock(&executionRegion);
    builder.setInsertionPointToEnd(executionBlock);
    
    // Create thread-local operation inside execution group
    SmallVector<Attribute> nameAttrs = {builder.getStringAttr("value")};
    SmallVector<Attribute> typeAttrs = {TypeAttr::get(builder.getI32Type())};
    auto nameArray = builder.getArrayAttr(nameAttrs);
    auto typeArray = builder.getArrayAttr(typeAttrs);
    auto stateMembers = subop::StateMembersAttr::get(&context, nameArray, typeArray);
    
    auto simpleStateType = subop::SimpleStateType::get(&context, stateMembers);
    auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
    auto createThreadLocalOp = builder.create<subop::CreateThreadLocalOp>(loc, threadLocalType);
    
    // Add execution group return
    builder.create<subop::ExecutionGroupReturnOp>(loc, ValueRange{});
    
    // Add terminator to main function
    builder.setInsertionPointToEnd(block);
    builder.create<func::ReturnOp>(loc);
    
    // Verify execution group was created
    EXPECT_TRUE(executionGroupOp);
    EXPECT_TRUE(createThreadLocalOp);
    
    // Verify proper termination
    auto mainTerminator = block->getTerminator();
    EXPECT_NE(mainTerminator, nullptr);
    EXPECT_TRUE(mainTerminator->hasTrait<OpTrait::IsTerminator>());
    
    auto execTerminator = executionBlock->getTerminator();
    EXPECT_NE(execTerminator, nullptr);
    EXPECT_TRUE(execTerminator->hasTrait<OpTrait::IsTerminator>());
    
    PGX_INFO("Execution group with thread-local test completed successfully");
    
    module.erase();
}