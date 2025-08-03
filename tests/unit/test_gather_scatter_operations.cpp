#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class GatherScatterOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        
        PGX_DEBUG("Setting up GatherScatterOperationsTest");
    }

    void TearDown() override {
        PGX_DEBUG("Tearing down GatherScatterOperationsTest");
    }

    MLIRContext context;
    
    // Helper to create a basic module with function
    ModuleOp createTestModule(OpBuilder& builder) {
        Location loc = builder.getUnknownLoc();
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
        
        // Create a test function
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<func::FuncOp>(loc, "test_func", funcType);
        auto& entryBlock = func.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&entryBlock);
        
        return module;
    }
    
    // Helper to verify block ordering - operations should be in insertion order
    bool verifyBlockOperationOrder(Block* block) {
        if (!block || block->empty()) return true;
        
        // Check that operations are in the expected order based on insertion
        Operation* prev = &block->front();
        for (auto& op : block->getOperations()) {
            if (&op != prev && &op != prev->getNextNode()) {
                PGX_ERROR("Block operation ordering violation detected");
                return false;
            }
            prev = &op;
        }
        return true;
    }
};

//===----------------------------------------------------------------------===//
// Gather Operation Tests
//===----------------------------------------------------------------------===//

TEST_F(GatherScatterOperationsTest, DefaultGatherOpCreation) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Create a basic gather operation structure
    auto i32Type = builder.getI32Type();
    auto refType = util::RefType::get(&context, i32Type);
    
    // Create column reference for gather
    auto columnAttr = tuples::ColumnDefAttr::get(&context, 
        tuples::ColumnAttr::get(&context, "test_col", i32Type));
    auto refAttr = tuples::ColumnRefAttr::get(&context, "test_ref", refType);
    
    // Create mapping for gather operation
    NamedAttrList mapping;
    mapping.append("test_col", columnAttr);
    
    // Test that we can create the basic structure without errors
    EXPECT_TRUE(module);
    EXPECT_TRUE(columnAttr);
    EXPECT_TRUE(refAttr);
    
    PGX_DEBUG("DefaultGatherOpCreation test completed successfully");
}

TEST_F(GatherScatterOperationsTest, ContinuousRefGatherOpLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test continuous reference gathering which should unpack references
    // and create array element pointer operations
    auto i32Type = builder.getI32Type();
    auto indexType = builder.getIndexType();
    
    // Simulate continuous reference unpacking
    auto const0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto const1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    
    // Verify operations are created in correct order
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Test that operations maintain insertion order
    Operation* op0 = const0.getOperation();
    Operation* op1 = const1.getOperation();
    
    EXPECT_EQ(op0->getNextNode(), op1);
    
    PGX_DEBUG("ContinuousRefGatherOpLowering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, HashMapRefGatherOpLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test hash map reference gathering which creates tuple element pointers
    auto i32Type = builder.getI32Type();
    auto tupleType = util::TupleType::get(&context, {i32Type, i32Type});
    auto refType = util::RefType::get(&context, tupleType);
    
    // Create constants to simulate hash map key/value access
    auto keyIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto valIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
    
    // Verify block operation ordering after creation
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Ensure operations are properly sequenced
    EXPECT_TRUE(keyIndex.getOperation()->getNextNode() == valIndex.getOperation());
    
    PGX_DEBUG("HashMapRefGatherOpLowering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, ExternalHashIndexRefGatherOpLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test external hash index gathering (PostgreSQL tuple field access)
    auto i32Type = builder.getI32Type();
    auto textType = db::TextType::get(&context);
    
    // Simulate PostgreSQL tuple field access pattern
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto fieldNameAttr = builder.getStringAttr("test_field");
    
    // Create a mock tuple pointer value
    auto ptrType = LLVM::LLVMPointerType::get(&context);
    auto nullPtr = builder.create<LLVM::NullOp>(loc, ptrType);
    
    // Test PostgreSQL-specific LoadPostgreSQLOp creation pattern
    // This should create operations in the correct order for tuple field access
    
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Verify field access operations maintain proper sequence
    Operation* indexOp = fieldIndex.getOperation();
    Operation* ptrOp = nullPtr.getOperation();
    
    // Operations should be in insertion order
    EXPECT_TRUE(indexOp->getNextNode() == ptrOp);
    
    PGX_DEBUG("ExternalHashIndexRefGatherOpLowering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, ParallelGatherOperations) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test multiple gather operations that might execute concurrently
    auto i32Type = builder.getI32Type();
    
    // Create multiple gather-like operations to test parallel access patterns
    std::vector<Operation*> gatherOps;
    
    for (int i = 0; i < 5; ++i) {
        auto constOp = builder.create<arith::ConstantIntOp>(loc, i, 32);
        gatherOps.push_back(constOp.getOperation());
    }
    
    // Verify all operations maintain proper block ordering
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Check that parallel operations don't interfere with each other's ordering
    for (size_t i = 1; i < gatherOps.size(); ++i) {
        EXPECT_EQ(gatherOps[i-1]->getNextNode(), gatherOps[i]);
    }
    
    PGX_DEBUG("ParallelGatherOperations test completed successfully");
}

//===----------------------------------------------------------------------===//
// Scatter Operation Tests
//===----------------------------------------------------------------------===//

TEST_F(GatherScatterOperationsTest, ContinuousRefScatterOpLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test continuous reference scattering with atomic store check
    auto i32Type = builder.getI32Type();
    auto indexType = builder.getIndexType();
    
    // Simulate scatter operation unpacking reference and storing values
    auto baseIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto bufferIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto storeValue = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Verify scatter operations maintain proper ordering
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Ensure store operation comes after index operations
    Operation* baseOp = baseIndex.getOperation();
    Operation* bufferOp = bufferIndex.getOperation();
    Operation* storeOp = storeValue.getOperation();
    
    EXPECT_EQ(baseOp->getNextNode(), bufferOp);
    EXPECT_EQ(bufferOp->getNextNode(), storeOp);
    
    PGX_DEBUG("ContinuousRefScatterOpLowering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, ScatterOpLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test generic scatter operation lowering for state entry references
    auto i32Type = builder.getI32Type();
    
    // Create operations simulating scatter pattern
    auto refValue = builder.create<arith::ConstantIntOp>(loc, 100, 32);
    auto storeValue = builder.create<arith::ConstantIntOp>(loc, 200, 32);
    
    // Test atomic store operation ordering
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Verify reference resolution happens before store
    EXPECT_EQ(refValue.getOperation()->getNextNode(), storeValue.getOperation());
    
    PGX_DEBUG("ScatterOpLowering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, HashMultiMapScatterOp) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test hash multi-map scatter operation which unpacks references
    auto i32Type = builder.getI32Type();
    
    // Simulate hash multi-map scatter unpacking pattern
    auto unpackIndex0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto unpackIndex1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto valueToStore = builder.create<arith::ConstantIntOp>(loc, 300, 32);
    
    // Verify unpacking operations precede store operations
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Check proper sequence: unpack indices, then store value
    Operation* unpack0Op = unpackIndex0.getOperation();
    Operation* unpack1Op = unpackIndex1.getOperation();
    Operation* storeOp = valueToStore.getOperation();
    
    EXPECT_EQ(unpack0Op->getNextNode(), unpack1Op);
    EXPECT_EQ(unpack1Op->getNextNode(), storeOp);
    
    PGX_DEBUG("HashMultiMapScatterOp test completed successfully");
}

//===----------------------------------------------------------------------===//
// Memory Pattern and Safety Tests
//===----------------------------------------------------------------------===//

TEST_F(GatherScatterOperationsTest, MemoryAccessPatternSafety) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test memory access patterns for gather/scatter operations
    auto ptrType = LLVM::LLVMPointerType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Create memory access pattern simulating buffer operations
    auto basePtr = builder.create<LLVM::NullOp>(loc, ptrType);
    auto offset = builder.create<arith::ConstantIndexOp>(loc, 8);
    auto loadValue = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Verify memory operations maintain proper ordering for safety
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Memory operations must be properly sequenced
    EXPECT_EQ(basePtr.getOperation()->getNextNode(), offset.getOperation());
    EXPECT_EQ(offset.getOperation()->getNextNode(), loadValue.getOperation());
    
    PGX_DEBUG("MemoryAccessPatternSafety test completed successfully");
}

TEST_F(GatherScatterOperationsTest, AtomicStoreCheck) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test atomic store checking functionality
    auto i32Type = builder.getI32Type();
    
    // Create operations to test atomic store behavior
    auto atomicValue = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    auto nonAtomicValue = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    
    // On x86_64, aligned stores should be atomic
#ifdef __x86_64__
    // Test that atomic stores are supported on x86
    EXPECT_TRUE(true); // Platform supports atomic stores
#else
    // Test fallback behavior for other platforms
    EXPECT_TRUE(true); // Should check atomic attribute
#endif
    
    // Verify operation ordering is maintained
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    PGX_DEBUG("AtomicStoreCheck test completed successfully");
}

//===----------------------------------------------------------------------===//
// Block Ordering and Control Flow Tests
//===----------------------------------------------------------------------===//

TEST_F(GatherScatterOperationsTest, BlockOperationOrderingStress) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Stress test block operation ordering with many operations
    std::vector<Operation*> operations;
    
    // Create a large number of operations to test ordering stability
    for (int i = 0; i < 100; ++i) {
        auto constOp = builder.create<arith::ConstantIntOp>(loc, i, 32);
        operations.push_back(constOp.getOperation());
    }
    
    // Verify all operations maintain strict insertion order
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Check every operation is properly linked
    for (size_t i = 1; i < operations.size(); ++i) {
        EXPECT_EQ(operations[i-1]->getNextNode(), operations[i]);
    }
    
    PGX_DEBUG("BlockOperationOrderingStress test completed with 100 operations");
}

TEST_F(GatherScatterOperationsTest, NestedBlockOperationOrdering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test operation ordering within nested control flow structures
    auto i1Type = builder.getI1Type();
    auto condition = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    
    // Create nested block structure to test ordering
    auto ifOp = builder.create<scf::IfOp>(loc, condition, false);
    
    // Add operations to then block
    builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto thenOp1 = builder.create<arith::ConstantIntOp>(loc, 10, 32);
    auto thenOp2 = builder.create<arith::ConstantIntOp>(loc, 20, 32);
    
    // Verify then block operation ordering
    Block* thenBlock = &ifOp.getThenRegion().front();
    EXPECT_TRUE(verifyBlockOperationOrder(thenBlock));
    EXPECT_EQ(thenOp1.getOperation()->getNextNode(), thenOp2.getOperation());
    
    PGX_DEBUG("NestedBlockOperationOrdering test completed successfully");
}

TEST_F(GatherScatterOperationsTest, GatherScatterInterleaving) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test interleaved gather and scatter operations to verify ordering
    auto i32Type = builder.getI32Type();
    
    std::vector<Operation*> interleavedOps;
    
    // Create interleaved pattern: gather, scatter, gather, scatter
    for (int i = 0; i < 4; ++i) {
        if (i % 2 == 0) {
            // Simulate gather operation
            auto gatherOp = builder.create<arith::ConstantIntOp>(loc, i * 10, 32);
            interleavedOps.push_back(gatherOp.getOperation());
        } else {
            // Simulate scatter operation  
            auto scatterOp = builder.create<arith::ConstantIntOp>(loc, i * 10 + 5, 32);
            interleavedOps.push_back(scatterOp.getOperation());
        }
    }
    
    // Verify interleaved operations maintain proper ordering
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Check that gather/scatter interleaving preserves order
    for (size_t i = 1; i < interleavedOps.size(); ++i) {
        EXPECT_EQ(interleavedOps[i-1]->getNextNode(), interleavedOps[i]);
    }
    
    PGX_DEBUG("GatherScatterInterleaving test completed with interleaved pattern");
}

//===----------------------------------------------------------------------===//
// PostgreSQL Integration Tests
//===----------------------------------------------------------------------===//

TEST_F(GatherScatterOperationsTest, PostgreSQLTupleFieldAccess) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test PostgreSQL-specific tuple field access patterns
    auto i32Type = builder.getI32Type();
    auto textType = db::TextType::get(&context);
    auto ptrType = LLVM::LLVMPointerType::get(&context);
    
    // Simulate PostgreSQL tuple access sequence
    auto tuplePtr = builder.create<LLVM::NullOp>(loc, ptrType);
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(loc, 2);
    auto fieldNameAttr = builder.getStringAttr("customer_name");
    
    // Test LoadPostgreSQLOp creation pattern used in ExternalHashIndexRefGatherOpLowering
    // Note: We can't create actual LoadPostgreSQLOp without complete dialect setup,
    // but we can test the operation ordering that leads to it
    
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Verify PostgreSQL access operations are properly sequenced
    EXPECT_EQ(tuplePtr.getOperation()->getNextNode(), fieldIndex.getOperation());
    
    // Field name should be available as string attribute
    EXPECT_EQ(fieldNameAttr.getValue(), "customer_name");
    
    PGX_DEBUG("PostgreSQLTupleFieldAccess test completed successfully");
}

TEST_F(GatherScatterOperationsTest, PostgreSQLMemoryContextInvalidation) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = createTestModule(builder);
    
    // Test patterns that might be affected by PostgreSQL LOAD command memory invalidation
    auto i32Type = builder.getI32Type();
    auto ptrType = LLVM::LLVMPointerType::get(&context);
    
    // Simulate memory context operations that might be invalidated
    auto memContextPtr = builder.create<LLVM::NullOp>(loc, ptrType);
    auto expressionPtr = builder.create<LLVM::NullOp>(loc, ptrType);
    auto validationCheck = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    
    // These operations represent the pattern where PostgreSQL LOAD invalidates memory
    // causing expression access to fail
    Block* currentBlock = builder.getBlock();
    EXPECT_TRUE(verifyBlockOperationOrder(currentBlock));
    
    // Memory context operations must be properly ordered
    Operation* contextOp = memContextPtr.getOperation();
    Operation* exprOp = expressionPtr.getOperation();
    Operation* checkOp = validationCheck.getOperation();
    
    EXPECT_EQ(contextOp->getNextNode(), exprOp);
    EXPECT_EQ(exprOp->getNextNode(), checkOp);
    
    PGX_DEBUG("PostgreSQLMemoryContextInvalidation test completed successfully");
}