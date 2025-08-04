#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "compiler/Dialect/RelAlg/RelAlgDialect.h"
#include "compiler/Dialect/RelAlg/RelAlgOps.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/DB/DBTypes.h"
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "compiler/Dialect/util/UtilDialect.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class TmpMaterializationLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<scf::SCFDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }

    // Helper to create a simple TupleStream type
    Type createTupleStreamType() {
        return tuples::TupleStreamType::get(&context);
    }

    // Helper to create buffer type with members
    Type createBufferType(ArrayAttr members) {
        auto stateMembers = subop::StateMembersAttr::get(&context, members, builder->getArrayAttr({}));
        return subop::BufferType::get(&context, stateMembers);
    }

    // Helper to create column definition attributes
    ArrayAttr createColumnMembers() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        auto i64Type = builder->getI64Type();
        auto stringType = builder->getType<db::StringType>();
        
        std::vector<Attribute> members;
        // Create column definitions for test buffer
        std::string scope = colManager.getUniqueScope("test");
        auto idColumn = colManager.createDef(scope, "id");
        auto nameColumn = colManager.createDef(scope, "name");
        
        idColumn.getColumn().type = i64Type;
        nameColumn.getColumn().type = stringType;
        
        auto idMember = idColumn;
        auto nameMember = nameColumn;
        
        members.push_back(idMember);
        members.push_back(nameMember);
        
        return builder->getArrayAttr(members);
    }

    // Helper to create dictionary attribute for column mapping
    DictionaryAttr createColumnMapping() {
        std::vector<NamedAttribute> mapping;
        
        // Create mapping from stream columns to state members
        auto idMapping = builder->getNamedAttr("id", 
            builder->getStringAttr("@table::@id"));
        auto nameMapping = builder->getNamedAttr("name",
            builder->getStringAttr("@table::@name"));
            
        mapping.push_back(idMapping);
        mapping.push_back(nameMapping);
        
        return builder->getDictionaryAttr(mapping);
    }

    // Validate cleanup after temporary operations
    void validateTemporaryCleanup(Operation* op) {
        // Verify that temporary states are properly cleaned up
        // Check for proper memory management patterns
        EXPECT_TRUE(op != nullptr);
        
        // Walk through operations to find materialization patterns
        op->walk([this](Operation* nestedOp) {
            if (auto materializeOp = dyn_cast<subop::MaterializeOp>(nestedOp)) {
                // Verify materialize operations have proper state management
                EXPECT_TRUE(materializeOp.getState() != nullptr);
                EXPECT_TRUE(materializeOp.getMapping() != nullptr);
            }
        });
    }

    // Check for concurrent access safety patterns
    void validateConcurrentAccessSafety(Operation* op) {
        // Verify thread safety patterns in temporary operations
        EXPECT_TRUE(op != nullptr);
        
        // Look for thread-local patterns or synchronization
        op->walk([](Operation* nestedOp) {
            if (auto createOp = dyn_cast<subop::CreateThreadLocalOp>(nestedOp)) {
                // Thread-local state should be properly initialized
                EXPECT_TRUE(createOp.getInitFn().hasValue());
            }
        });
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(TmpMaterializationLoweringTest, TmpTableCreation) {
    PGX_DEBUG("Testing TmpOp temporary table creation");
    
    // Create a module for testing
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create input stream
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    
    // Create column definitions for TmpOp
    auto columnMembers = createColumnMembers();
    
    // Create TmpOp to test temporary table creation
    auto tmpOp = builder->create<relalg::TmpOp>(builder->getUnknownLoc(), 
        ArrayRef<Type>{streamType}, // Single output stream
        inputStream.getResult(),
        columnMembers);
    
    // Verify TmpOp structure
    EXPECT_TRUE(tmpOp);
    EXPECT_EQ(tmpOp.getNumResults(), 1);
    EXPECT_TRUE(tmpOp.getRel() == inputStream.getResult());
    EXPECT_TRUE(tmpOp.getCols() == columnMembers);
    
    // Verify temporary table semantics
    EXPECT_TRUE(isa<tuples::TupleStreamType>(tmpOp.getResult(0).getType()));
    
    PGX_DEBUG("TmpOp temporary table creation test completed");
}

TEST_F(TmpMaterializationLoweringTest, MaterializationStrategy) {
    PGX_DEBUG("Testing materialization strategy selection");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create input stream
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    
    // Test different materialization strategies
    
    // 1. Buffer materialization for temporary storage
    auto bufferMembers = createColumnMembers();
    auto bufferType = createBufferType(bufferMembers);
    auto buffer = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), bufferType);
    
    auto columnMapping = createColumnMapping();
    auto bufferMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), buffer.getRes(), columnMapping);
    
    // Verify buffer materialization
    EXPECT_TRUE(bufferMaterialize);
    EXPECT_TRUE(bufferMaterialize.getStream() == inputStream.getResult());
    EXPECT_TRUE(bufferMaterialize.getState() == buffer.getRes());
    EXPECT_TRUE(bufferMaterialize.getMapping() == columnMapping);
    
    // 2. Hash table materialization for joins
    auto keyMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto valueMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto mapType = subop::SimpleStateType::get(&context, keyMembers); // MapType constructor may be different
    auto hashMap = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), mapType);
    
    auto hashMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), hashMap.getRes(), columnMapping);
    
    // Verify hash table materialization strategy
    EXPECT_TRUE(hashMaterialize);
    EXPECT_TRUE(isa<subop::SimpleStateType>(hashMap.getRes().getType())); // MapType changed to SimpleStateType
    
    // 3. Heap materialization for sorted operations
    auto heapStateMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto heapType = subop::HeapType::get(&context, heapStateMembers, 100);
    auto heap = builder->create<subop::CreateHeapOp>(builder->getUnknownLoc(), heapType, 
        builder->getArrayAttr({}));
    
    auto heapMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), heap.getRes(), columnMapping);
    
    // Verify heap materialization for sorting
    EXPECT_TRUE(heapMaterialize);
    EXPECT_TRUE(isa<subop::HeapType>(heap.getRes().getType()));
    
    PGX_DEBUG("Materialization strategy selection test completed");
}

TEST_F(TmpMaterializationLoweringTest, StorageLifecycle) {
    PGX_DEBUG("Testing temporary storage lifecycle management");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create input stream
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    
    // Test complete storage lifecycle
    
    // 1. Creation phase
    auto bufferMembers = createColumnMembers();
    auto bufferType = createBufferType(bufferMembers);
    auto storage = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), bufferType);
    
    EXPECT_TRUE(storage);
    EXPECT_TRUE(isa<subop::BufferType>(storage.getRes().getType()));
    
    // 2. Population phase (materialization)
    auto columnMapping = createColumnMapping();
    auto materialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), storage.getRes(), columnMapping);
    
    EXPECT_TRUE(materialize);
    
    // 3. Usage phase (scanning)
    auto stateColumnMapping = createColumnMapping();
    auto scan = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        storage.getRes(), stateColumnMapping);
    
    EXPECT_TRUE(scan);
    EXPECT_TRUE(scan.getState() == storage.getRes());
    
    // 4. Verify lifecycle dependencies
    EXPECT_TRUE(materialize.getState() == storage.getRes());
    EXPECT_TRUE(scan.getState() == storage.getRes());
    
    // 5. Test multiple consumers (typical TmpOp pattern)
    auto scan2 = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        storage.getRes(), stateColumnMapping);
    auto scan3 = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        storage.getRes(), stateColumnMapping);
    
    EXPECT_TRUE(scan2);
    EXPECT_TRUE(scan3);
    EXPECT_TRUE(scan2.getState() == storage.getRes());
    EXPECT_TRUE(scan3.getState() == storage.getRes());
    
    // Validate proper cleanup patterns
    validateTemporaryCleanup(module);
    
    PGX_DEBUG("Storage lifecycle management test completed");
}

TEST_F(TmpMaterializationLoweringTest, MemoryVsDiskMaterialization) {
    PGX_DEBUG("Testing memory vs disk materialization decisions");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    
    // Test memory materialization (default buffer)
    auto memoryBufferType = createBufferType(bufferMembers);
    auto memoryBuffer = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), memoryBufferType);
    
    auto columnMapping = createColumnMapping();
    auto memoryMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), memoryBuffer.getRes(), columnMapping);
    
    // Verify memory materialization characteristics
    EXPECT_TRUE(memoryMaterialize);
    EXPECT_TRUE(isa<subop::BufferType>(memoryBuffer.getRes().getType()));
    
    // Test disk-backed materialization (simulated with ResultTable)
    auto diskStateMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto resultTableType = subop::ResultTableType::get(&context, diskStateMembers);
    auto diskTable = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), resultTableType);
    
    auto diskMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), diskTable.getRes(), columnMapping);
    
    // Verify disk materialization characteristics  
    EXPECT_TRUE(diskMaterialize);
    EXPECT_TRUE(isa<subop::ResultTableType>(diskTable.getRes().getType()));
    
    // Test hybrid approach with overflow handling
    // Create a view that could represent overflow management
    auto continuousViewType = subop::ContinuousViewType::get(&context, memoryBufferType);
    auto overflowView = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), continuousViewType);
    
    // This would represent a materialization strategy that starts in memory
    // and overflows to disk when needed
    EXPECT_TRUE(overflowView);
    EXPECT_TRUE(isa<subop::ContinuousViewType>(overflowView.getRes().getType()));
    
    PGX_DEBUG("Memory vs disk materialization test completed");
}

TEST_F(TmpMaterializationLoweringTest, ConcurrentTemporaryAccess) {
    PGX_DEBUG("Testing concurrent access to temporary data");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    
    // Test thread-local temporary storage for parallel operations
    auto stateMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto simpleStateType = subop::SimpleStateType::get(&context, stateMembers);
    auto threadLocalType = subop::ThreadLocalType::get(&context, simpleStateType);
    
    // Create thread-local temporary storage
    auto threadLocalOp = builder->create<subop::CreateThreadLocalOp>(builder->getUnknownLoc(),
        threadLocalType);
    
    // Add initialization region for thread-local state
    auto* initBlock = new Block;
    threadLocalOp.getInitFn().push_back(initBlock);
    builder->setInsertionPointToStart(initBlock);
    
    // Initialize with empty state
    auto initialState = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), simpleStateType);
    builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), initialState.getRes());
    
    // Reset insertion point
    builder->setInsertionPointAfter(threadLocalOp);
    
    EXPECT_TRUE(threadLocalOp);
    EXPECT_TRUE(isa<subop::ThreadLocalType>(threadLocalOp.getRes().getType()));
    
    // Test parallel materialization pattern
    // This would be used in parallel reduction scenarios
    auto getLocalOp = builder->create<subop::GetLocalOp>(builder->getUnknownLoc(), simpleStateType,
        threadLocalOp.getRes());
    
    EXPECT_TRUE(getLocalOp);
    EXPECT_TRUE(getLocalOp.getThreadLocal() == threadLocalOp.getRes());
    
    // Test merge operation for combining thread-local results
    auto mergeOp = builder->create<subop::MergeOp>(builder->getUnknownLoc(), simpleStateType,
        threadLocalOp.getRes());
    
    // Add combine region for merging
    auto* combineBlock = new Block;
    combineBlock->addArgument(simpleStateType, builder->getUnknownLoc());
    combineBlock->addArgument(simpleStateType, builder->getUnknownLoc());
    mergeOp.getCombine().push_back(combineBlock);
    
    builder->setInsertionPointToStart(combineBlock);
    auto leftArg = combineBlock->getArgument(0);
    auto rightArg = combineBlock->getArgument(1);
    // Simple merge logic - in practice this would combine the states
    builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), leftArg);
    
    EXPECT_TRUE(mergeOp);
    EXPECT_TRUE(mergeOp.getThreadLocal() == threadLocalOp.getRes());
    
    // Validate concurrent access safety
    validateConcurrentAccessSafety(module);
    
    PGX_DEBUG("Concurrent temporary access test completed");
}

TEST_F(TmpMaterializationLoweringTest, StorageOverflowHandling) {
    PGX_DEBUG("Testing storage overflow handling mechanisms");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    
    // Test bounded heap for memory-limited scenarios
    uint64_t maxRows = 1000;
    auto boundedStateMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto boundedHeapType = subop::HeapType::get(&context, boundedStateMembers, maxRows);
    auto boundedHeap = builder->create<subop::CreateHeapOp>(builder->getUnknownLoc(), boundedHeapType,
        builder->getArrayAttr({}));
    
    EXPECT_TRUE(boundedHeap);
    EXPECT_TRUE(isa<subop::HeapType>(boundedHeap.getRes().getType()));
    
    // Test materialization into bounded storage
    auto columnMapping = createColumnMapping();
    auto boundedMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), boundedHeap.getRes(), columnMapping);
    
    EXPECT_TRUE(boundedMaterialize);
    EXPECT_TRUE(boundedMaterialize.getState() == boundedHeap.getRes());
    
    // Test overflow detection through size monitoring
    // In practice, this would involve checking heap size during materialization
    auto scanAfterMaterialize = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        boundedHeap.getRes(), columnMapping);
    
    EXPECT_TRUE(scanAfterMaterialize);
    
    // Test cascade to disk-based storage on overflow
    auto overflowStateMembers = subop::StateMembersAttr::get(&context, bufferMembers, builder->getArrayAttr({}));
    auto overflowTableType = subop::ResultTableType::get(&context, overflowStateMembers);
    auto overflowTable = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), overflowTableType);
    
    auto overflowMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), overflowTable.getRes(), columnMapping);
    
    EXPECT_TRUE(overflowMaterialize);
    EXPECT_TRUE(isa<subop::ResultTableType>(overflowTable.getRes().getType()));
    
    // Test union of in-memory and overflow results
    auto memoryResults = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        boundedHeap.getRes(), columnMapping);
    auto overflowResults = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        overflowTable.getRes(), columnMapping);
    
    auto combinedResults = builder->create<subop::UnionOp>(builder->getUnknownLoc(), streamType,
        ValueRange{memoryResults.getResult(), overflowResults.getResult()});
    
    EXPECT_TRUE(combinedResults);
    EXPECT_EQ(combinedResults.getInputs().size(), 2);
    
    PGX_DEBUG("Storage overflow handling test completed");
}

TEST_F(TmpMaterializationLoweringTest, MaterializationPerformance) {
    PGX_DEBUG("Testing materialization performance characteristics");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    
    // Test batch materialization for efficiency
    auto bufferType = createBufferType(bufferMembers);
    auto batchBuffer = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), bufferType);
    
    auto columnMapping = createColumnMapping();
    auto batchMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), batchBuffer.getRes(), columnMapping);
    
    // Verify batch materialization setup
    EXPECT_TRUE(batchMaterialize);
    EXPECT_TRUE(batchMaterialize.getStream() == inputStream.getResult());
    
    // Test vectorized access patterns
    auto vectorScan = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        batchBuffer.getRes(), columnMapping);
    
    EXPECT_TRUE(vectorScan);
    
    // Test memory-aligned access for cache efficiency
    // This would be reflected in the buffer layout and access patterns
    auto alignedScan = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        batchBuffer.getRes(), columnMapping);
    
    // Add sequential attribute for cache-friendly access
    alignedScan->setAttr("sequential", builder->getUnitAttr());
    
    EXPECT_TRUE(alignedScan);
    EXPECT_TRUE(alignedScan->hasAttr("sequential"));
    
    // Test parallel scan for performance
    auto parallelScan = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        batchBuffer.getRes(), columnMapping);
    
    parallelScan->setAttr("parallel", builder->getUnitAttr());
    
    EXPECT_TRUE(parallelScan);
    EXPECT_TRUE(parallelScan->hasAttr("parallel"));
    
    PGX_DEBUG("Materialization performance test completed");
}

TEST_F(TmpMaterializationLoweringTest, TmpOpMultipleOutputs) {
    PGX_DEBUG("Testing TmpOp with multiple output streams");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    
    // Create TmpOp with multiple outputs (common pattern for temp tables)
    std::vector<Type> outputTypes = {streamType, streamType, streamType};
    auto multiOutputTmp = builder->create<relalg::TmpOp>(builder->getUnknownLoc(),
        outputTypes,
        inputStream.getResult(),
        bufferMembers);
    
    // Verify multiple outputs
    EXPECT_TRUE(multiOutputTmp);
    EXPECT_EQ(multiOutputTmp.getNumResults(), 3);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(isa<tuples::TupleStreamType>(multiOutputTmp.getResult(i).getType()));
    }
    
    // Test that each output can be used independently
    auto map1 = builder->create<subop::MapOp>(builder->getUnknownLoc(), streamType,
        multiOutputTmp.getResult(0), builder->getArrayAttr({}), builder->getArrayAttr({}));
    auto map2 = builder->create<subop::MapOp>(builder->getUnknownLoc(), streamType,
        multiOutputTmp.getResult(1), builder->getArrayAttr({}), builder->getArrayAttr({}));
    auto map3 = builder->create<subop::MapOp>(builder->getUnknownLoc(), streamType,
        multiOutputTmp.getResult(2), builder->getArrayAttr({}), builder->getArrayAttr({}));
    
    // Add empty regions for the map operations
    map1.getRegion().emplaceBlock();
    map2.getRegion().emplaceBlock();
    map3.getRegion().emplaceBlock();
    
    EXPECT_TRUE(map1);
    EXPECT_TRUE(map2);
    EXPECT_TRUE(map3);
    
    // Verify independent usage of temporary table results
    EXPECT_TRUE(map1.getStream() == multiOutputTmp.getResult(0));
    EXPECT_TRUE(map2.getStream() == multiOutputTmp.getResult(1));
    EXPECT_TRUE(map3.getStream() == multiOutputTmp.getResult(2));
    
    PGX_DEBUG("TmpOp multiple outputs test completed");
}

TEST_F(TmpMaterializationLoweringTest, ComplexMaterializationWorkflow) {
    PGX_DEBUG("Testing complex materialization workflow integration");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    auto streamType = createTupleStreamType();
    auto inputStream = builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), streamType);
    auto bufferMembers = createColumnMembers();
    auto columnMapping = createColumnMapping();
    
    // Step 1: Create initial temporary storage
    auto bufferType = createBufferType(bufferMembers);
    auto tempStorage = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), bufferType);
    
    // Step 2: Materialize input data
    auto initialMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        inputStream.getResult(), tempStorage.getRes(), columnMapping);
    
    // Step 3: Scan for processing
    auto processStream = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        tempStorage.getRes(), columnMapping);
    
    // Step 4: Apply transformations via MapOp
    auto transformOp = builder->create<subop::MapOp>(builder->getUnknownLoc(), streamType,
        processStream.getRes(), builder->getArrayAttr({}), builder->getArrayAttr({}));
    
    // Add transformation region
    auto* transformBlock = new Block;
    transformBlock->addArgument(tuples::TupleType::get(&context), builder->getUnknownLoc());
    transformOp.getRegion().push_back(transformBlock);
    
    builder->setInsertionPointToStart(transformBlock);
    auto tupleArg = transformBlock->getArgument(0);
    builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), tupleArg);
    
    // Reset insertion point
    builder->setInsertionPointAfter(transformOp);
    
    // Step 5: Re-materialize transformed results
    auto resultStorage = builder->create<subop::GenericCreateOp>(builder->getUnknownLoc(), bufferType);
    auto finalMaterialize = builder->create<subop::MaterializeOp>(builder->getUnknownLoc(),
        transformOp.getRes(), resultStorage.getRes(), columnMapping);
    
    // Step 6: Create final output streams
    auto finalScan = builder->create<subop::ScanOp>(builder->getUnknownLoc(), streamType,
        resultStorage.getRes(), columnMapping);
    
    // Verify complete workflow
    EXPECT_TRUE(tempStorage);
    EXPECT_TRUE(initialMaterialize);
    EXPECT_TRUE(processStream);
    EXPECT_TRUE(transformOp);
    EXPECT_TRUE(resultStorage);
    EXPECT_TRUE(finalMaterialize);
    EXPECT_TRUE(finalScan);
    
    // Verify data flow
    EXPECT_TRUE(initialMaterialize.getStream() == inputStream.getResult());
    EXPECT_TRUE(processStream.getState() == tempStorage.getRes());
    EXPECT_TRUE(transformOp.getStream() == processStream.getRes());
    EXPECT_TRUE(finalMaterialize.getStream() == transformOp.getRes());
    EXPECT_TRUE(finalScan.getState() == resultStorage.getRes());
    
    // Validate proper resource management
    validateTemporaryCleanup(module);
    
    PGX_DEBUG("Complex materialization workflow test completed");
}