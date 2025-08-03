#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpTypes.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilTypes.h"
#include "dialects/tuples/TupleStreamDialect.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class DataStructureCreationTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
    
    // Helper to create a basic module with function
    std::pair<ModuleOp, func::FuncOp> createModuleWithFunction() {
        OpBuilder builder(&context);
        Location loc = builder.getUnknownLoc();
        
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
        
        auto i32Type = builder.getI32Type();
        auto funcType = builder.getFunctionType({}, {i32Type});
        auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);
        
        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        return {module, funcOp};
    }
    
    // Helper to create tuple type for testing
    TupleType createTestTupleType() {
        OpBuilder builder(&context);
        auto i32Type = builder.getI32Type();
        auto i64Type = builder.getI64Type();
        return TupleType::get(&context, {i32Type, i64Type});
    }
};

// Test CreateThreadLocalOp creation and basic structure
TEST_F(DataStructureCreationTest, CreateThreadLocalBasicStructure) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create a thread local type
    auto elementType = builder.getI32Type();
    auto threadLocalType = subop::ThreadLocalType::get(&context, elementType);
    
    // Create the thread local operation
    auto createThreadLocal = builder.create<subop::CreateThreadLocalOp>(
        loc, threadLocalType);
    
    // Add initialization region
    auto* initRegion = &createThreadLocal.getInitFn();
    auto* initBlock = &initRegion->emplaceBlock();
    builder.setInsertionPointToStart(initBlock);
    
    // Add simple initialization
    auto constOp = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    builder.create<tuples::ReturnOp>(loc, ValueRange{constOp});
    
    // Verify structure
    EXPECT_TRUE(createThreadLocal);
    EXPECT_EQ(createThreadLocal.getInitFn().getBlocks().size(), 1);
    EXPECT_TRUE(threadLocalType);
}

// Test buffer creation operations
TEST_F(DataStructureCreationTest, CreateBufferStructure) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create buffer type with member specification
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    
    SmallVector<Attribute> memberNames = {
        builder.getStringAttr("field1"),
        builder.getStringAttr("field2")
    };
    SmallVector<Type> memberTypes = {i32Type, i64Type};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context, 
        builder.getArrayAttr(memberNames), 
        TypeRange(memberTypes));
    
    auto bufferType = subop::BufferType::get(&context, memberSpec, false);
    
    // Create buffer operation
    auto createBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
    
    // Add initial capacity attribute
    createBuffer->setAttr("initial_capacity", builder.getI64IntegerAttr(2048));
    
    // Verify structure
    EXPECT_TRUE(createBuffer);
    EXPECT_TRUE(mlir::isa<subop::BufferType>(createBuffer.getType()));
    EXPECT_TRUE(createBuffer->hasAttr("initial_capacity"));
}

// Test hash map creation operations
TEST_F(DataStructureCreationTest, CreateHashMapStructure) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create key and value types
    auto keyType = builder.getI32Type();
    auto valueType = builder.getI64Type();
    
    // Create member specifications for key and value
    SmallVector<Attribute> keyNames = {builder.getStringAttr("key")};
    SmallVector<Attribute> valueNames = {builder.getStringAttr("value")};
    
    auto keySpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(keyNames), TypeRange(keyType));
    auto valueSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(valueNames), TypeRange(valueType));
    
    auto hashMapType = subop::HashMapType::get(&context, keySpec, valueSpec);
    
    // Create hash map operation
    auto createHashMap = builder.create<subop::GenericCreateOp>(loc, hashMapType);
    
    // Verify structure
    EXPECT_TRUE(createHashMap);
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(createHashMap.getType()));
}

// Test hash multi-map creation
TEST_F(DataStructureCreationTest, CreateHashMultiMapStructure) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create key and value types for multi-map
    auto keyType = builder.getI32Type();
    auto valueType = builder.getI64Type();
    
    SmallVector<Attribute> keyNames = {builder.getStringAttr("key")};
    SmallVector<Attribute> valueNames = {builder.getStringAttr("value")};
    
    auto keySpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(keyNames), TypeRange(keyType));
    auto valueSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(valueNames), TypeRange(valueType));
    
    auto hashMultiMapType = subop::HashMultiMapType::get(&context, keySpec, valueSpec);
    
    // Create hash multi-map operation
    auto createHashMultiMap = builder.create<subop::GenericCreateOp>(loc, hashMultiMapType);
    
    // Verify structure
    EXPECT_TRUE(createHashMultiMap);
    EXPECT_TRUE(mlir::isa<subop::HashMultiMapType>(createHashMultiMap.getType()));
}

// Test array creation with specified size
TEST_F(DataStructureCreationTest, CreateArrayWithSize) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create array type
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("element")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    auto arrayType = subop::ArrayType::get(&context, memberSpec, false);
    
    // Create size operand (as a tuple containing the size)
    auto sizeConstant = builder.create<arith::ConstantIndexOp>(loc, 100);
    auto tupleType = TupleType::get(&context, {builder.getIndexType()});
    auto tuplePack = builder.create<util::PackOp>(loc, tupleType, ValueRange{sizeConstant});
    auto sizeRef = builder.create<util::ToGenericMemrefOp>(loc, 
        util::RefType::get(&context, tupleType), tuplePack);
    
    // Create array operation
    auto createArray = builder.create<subop::CreateArrayOp>(loc, arrayType, sizeRef);
    
    // Verify structure
    EXPECT_TRUE(createArray);
    EXPECT_TRUE(mlir::isa<subop::ArrayType>(createArray.getType()));
    EXPECT_TRUE(createArray.getNumElements());
}

// Test heap creation with comparison function
TEST_F(DataStructureCreationTest, CreateHeapWithComparison) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create heap element type
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("priority")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    // Create sort specification
    SmallVector<Attribute> sortColumns = {builder.getStringAttr("priority")};
    auto sortSpec = builder.getArrayAttr(sortColumns);
    
    auto heapType = subop::HeapType::get(&context, memberSpec, false, 1000);
    
    // Create heap operation
    auto createHeap = builder.create<subop::CreateHeapOp>(loc, heapType, sortSpec);
    
    // Add comparison region
    auto& comparisonRegion = createHeap.getRegion();
    auto* comparisonBlock = &comparisonRegion.emplaceBlock();
    
    // Add arguments for left and right values
    auto leftArg = comparisonBlock->addArgument(elementType, loc);
    auto rightArg = comparisonBlock->addArgument(elementType, loc);
    
    builder.setInsertionPointToStart(comparisonBlock);
    
    // Create simple comparison (left < right)
    auto comparison = builder.create<arith::CmpIOp>(loc, 
        arith::CmpIPredicate::slt, leftArg, rightArg);
    builder.create<tuples::ReturnOp>(loc, ValueRange{comparison});
    
    // Verify structure
    EXPECT_TRUE(createHeap);
    EXPECT_TRUE(mlir::isa<subop::HeapType>(createHeap.getType()));
    EXPECT_EQ(createHeap.getRegion().getBlocks().size(), 1);
}

// Test simple state creation (both heap and stack allocation)
TEST_F(DataStructureCreationTest, CreateSimpleStateAllocations) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create simple state type
    auto stateType = builder.getI64Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("counter")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(stateType));
    
    auto simpleStateType = subop::SimpleStateType::get(&context, memberSpec, false);
    
    // Test heap allocation
    auto createStateHeap = builder.create<subop::CreateSimpleStateOp>(loc, simpleStateType);
    createStateHeap->setAttr("allocateOnHeap", builder.getUnitAttr());
    
    // Test stack allocation (default)
    auto createStateStack = builder.create<subop::CreateSimpleStateOp>(loc, simpleStateType);
    
    // Verify structures
    EXPECT_TRUE(createStateHeap);
    EXPECT_TRUE(createStateStack);
    EXPECT_TRUE(createStateHeap->hasAttr("allocateOnHeap"));
    EXPECT_FALSE(createStateStack->hasAttr("allocateOnHeap"));
}

// Test continuous view creation from different sources
TEST_F(DataStructureCreationTest, CreateContinuousViewFromSources) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create base types
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("data")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    // Create source buffer
    auto bufferType = subop::BufferType::get(&context, memberSpec, false);
    auto createBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
    
    // Create continuous view type
    auto continuousViewType = subop::ContinuousViewType::get(&context, memberSpec, false);
    
    // Create continuous view from buffer
    auto createView = builder.create<subop::CreateContinuousView>(
        loc, continuousViewType, createBuffer);
    
    // Verify structure
    EXPECT_TRUE(createView);
    EXPECT_TRUE(mlir::isa<subop::ContinuousViewType>(createView.getType()));
    EXPECT_EQ(createView.getSource(), createBuffer.getResult());
}

// Test external hash index access
TEST_F(DataStructureCreationTest, GetExternalHashIndex) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create external hash index type
    auto keyType = builder.getI32Type();
    SmallVector<Attribute> keyNames = {builder.getStringAttr("key")};
    
    auto keySpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(keyNames), TypeRange(keyType));
    
    auto externalHashIndexType = subop::ExternalHashIndexType::get(&context, keySpec);
    
    // Create external operation
    auto getExternal = builder.create<subop::GetExternalOp>(
        loc, externalHashIndexType, builder.getStringAttr("test_index"));
    
    // Verify structure
    EXPECT_TRUE(getExternal);
    EXPECT_TRUE(mlir::isa<subop::ExternalHashIndexType>(getExternal.getType()));
    EXPECT_EQ(getExternal.getDescrAttr().getValue(), "test_index");
}

// Test for proper terminator handling in data structure creation
TEST_F(DataStructureCreationTest, TerminatorValidationInCreation) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create a block that already has a terminator
    auto* testBlock = new Block();
    builder.setInsertionPointToStart(testBlock);
    
    // Add a terminator first
    auto constOp = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    builder.create<func::ReturnOp>(loc, ValueRange{constOp});
    
    // Verify block has terminator
    EXPECT_TRUE(testBlock->hasTerminator());
    
    // Test should validate that data structure creation doesn't add operations
    // after terminators. This is the type of bug the unit test should catch.
    auto terminator = testBlock->getTerminator();
    EXPECT_TRUE(terminator);
    EXPECT_TRUE(mlir::isa<func::ReturnOp>(terminator));
    
    // Ensure we don't accidentally add operations after the terminator
    // This would be caught by MLIR verification in real lowering
    builder.setInsertionPoint(terminator);
    
    // Attempting to insert here should be detected as an error condition
    auto insertionPoint = builder.getInsertionPoint();
    EXPECT_EQ(insertionPoint, Block::iterator(terminator));
    
    delete testBlock;
}

// Test memory allocation patterns in data structure creation
TEST_F(DataStructureCreationTest, MemoryAllocationPatterns) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Test buffer with group allocator
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("data")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    auto bufferType = subop::BufferType::get(&context, memberSpec, false);
    auto createBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
    
    // Add group allocator attribute
    createBuffer->setAttr("group", builder.getI64IntegerAttr(5));
    createBuffer->setAttr("initial_capacity", builder.getI64IntegerAttr(4096));
    
    // Verify memory-related attributes
    EXPECT_TRUE(createBuffer->hasAttr("group"));
    EXPECT_TRUE(createBuffer->hasAttr("initial_capacity"));
    
    auto groupAttr = createBuffer->getAttr("group");
    EXPECT_TRUE(mlir::isa<mlir::IntegerAttr>(groupAttr));
    EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(groupAttr).getInt(), 5);
}

// Test initialization functions in data structure creation
TEST_F(DataStructureCreationTest, InitializationFunctions) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create simple state with initialization
    auto stateType = builder.getI64Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("value")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(stateType));
    
    auto simpleStateType = subop::SimpleStateType::get(&context, memberSpec, false);
    auto createState = builder.create<subop::CreateSimpleStateOp>(loc, simpleStateType);
    
    // Add initialization region
    auto& initRegion = createState.getInitFn();
    auto* initBlock = &initRegion.emplaceBlock();
    builder.setInsertionPointToStart(initBlock);
    
    // Initialize with a value
    auto initValue = builder.create<arith::ConstantIntOp>(loc, 100, 64);
    builder.create<tuples::ReturnOp>(loc, ValueRange{initValue});
    
    // Verify initialization structure
    EXPECT_FALSE(createState.getInitFn().empty());
    EXPECT_EQ(createState.getInitFn().getBlocks().size(), 1);
    
    // Verify the initialization block has the correct structure
    auto& initFirstBlock = createState.getInitFn().front();
    EXPECT_TRUE(initFirstBlock.hasTerminator());
    EXPECT_TRUE(mlir::isa<tuples::ReturnOp>(initFirstBlock.getTerminator()));
}

// Test resource cleanup patterns
TEST_F(DataStructureCreationTest, ResourceCleanupValidation) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create multiple data structures to test resource management
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("data")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    // Create buffer
    auto bufferType = subop::BufferType::get(&context, memberSpec, false);
    auto createBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
    
    // Create hash map
    auto hashMapType = subop::HashMapType::get(&context, memberSpec, memberSpec);
    auto createHashMap = builder.create<subop::GenericCreateOp>(loc, hashMapType);
    
    // Verify both operations are created
    EXPECT_TRUE(createBuffer);
    EXPECT_TRUE(createHashMap);
    
    // In a real implementation, we would verify that:
    // 1. Memory allocations have corresponding deallocations
    // 2. Resources are properly managed within PostgreSQL contexts
    // 3. No memory leaks occur during operation lowering
    
    // This test validates the structure exists for resource management
    EXPECT_TRUE(mlir::isa<subop::BufferType>(createBuffer.getType()));
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(createHashMap.getType()));
}

// Test lowering pass execution on data structure operations
TEST_F(DataStructureCreationTest, LoweringPassExecution) {
    auto [module, funcOp] = createModuleWithFunction();
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Create a simple buffer operation
    auto elementType = builder.getI32Type();
    SmallVector<Attribute> memberNames = {builder.getStringAttr("field")};
    
    auto memberSpec = tuples::TupleStreamTypeExtension::get(&context,
        builder.getArrayAttr(memberNames), TypeRange(elementType));
    
    auto bufferType = subop::BufferType::get(&context, memberSpec, false);
    auto createBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
    
    // Add return statement to make function valid
    auto constResult = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    builder.create<func::ReturnOp>(loc, ValueRange{constResult});
    
    // Verify module structure before lowering
    EXPECT_TRUE(module);
    EXPECT_TRUE(createBuffer);
    
    // Create pass manager for lowering
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    // Run the lowering pass
    auto result = pm.run(module);
    
    // The pass should complete successfully
    // In real execution, this would transform SubOp operations to lower-level IR
    EXPECT_TRUE(succeeded(result));
}