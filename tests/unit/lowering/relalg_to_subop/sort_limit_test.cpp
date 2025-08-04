#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "compiler/Dialect/RelAlg/RelAlgDialect.h"
#include "compiler/Dialect/RelAlg/RelAlgOps.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/DB/DBDialect.h"
#include "compiler/Dialect/DB/DBTypes.h"
#include "compiler/Dialect/util/UtilDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"

// TODO Phase 5: Include the actual lowering pass headers when implemented
// #include "compiler/Dialect/RelAlg/RelAlgToSubOpPass.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class SortLimitLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }

    // Create a test module with RelAlg operations for lowering
    ModuleOp createTestModule() {
        auto module = ModuleOp::create(builder->getUnknownLoc());
        builder->setInsertionPointToEnd(module.getBody());
        return module;
    }

    // Create a test table relation
    Value createTestTableScan() {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        return builder->create<relalg::BaseTableOp>(builder->getUnknownLoc(), tupleStreamType, 
            builder->getStringAttr("test_table"), 
            builder->getDictionaryAttr({})); // Empty columns for testing
    }

    // Create test column references
    tuples::ColumnRefAttr createColumnRef(StringRef name, Type type) {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        return colManager.createRef("test_table", name);
    }

    // Create sort specification attributes
    relalg::SortSpecificationAttr createSortSpec(tuples::ColumnRefAttr columnRef, 
                                                  relalg::SortSpec order) {
        return relalg::SortSpecificationAttr::get(&context, columnRef, order);
    }

    // Execute lowering passes (TODO Phase 5: Implement actual pass pipeline)
    LogicalResult runLoweringPasses(ModuleOp module) {
        PassManager pm(&context);
        // TODO Phase 5: Add RelAlg → SubOp lowering passes
        // pm.addPass(createRelAlgToSubOpPass());
        
        // For now, verify the RelAlg operations exist
        return success();
    }

    // Verify that a lowered operation contains specific SubOp types
    void verifyContainsSubOpType(Operation* op, StringRef opName) {
        bool found = false;
        op->walk([&](Operation* walkOp) {
            if (walkOp->getName().getStringRef() == opName) {
                found = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        EXPECT_TRUE(found) << "Expected to find " << opName.str() << " in lowered operation";
    }

    // Count operations of a specific type
    int countSubOpType(Operation* op, StringRef opName) {
        int count = 0;
        op->walk([&](Operation* walkOp) {
            if (walkOp->getName().getStringRef() == opName) {
                count++;
            }
            return WalkResult::advance();
        });
        return count;
    }

    // Verify RelAlg operation exists in module
    void verifyRelAlgOpExists(ModuleOp module, StringRef opName) {
        bool found = false;
        module.walk([&](Operation* op) {
            if (op->getName().getStringRef() == opName) {
                found = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        EXPECT_TRUE(found) << "Expected to find RelAlg operation " << opName.str();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

//===----------------------------------------------------------------------===//
// LimitOp Lowering Tests
//===----------------------------------------------------------------------===//

TEST_F(SortLimitLoweringTest, LimitOperationLowering) {
    // Test that LimitOp lowers to CreateHeapOp + MaterializeOp + ScanOp
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto limitOp = builder->create<relalg::LimitOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(10), 
        tableScan);
    
    // Verify RelAlg operation was created
    verifyRelAlgOpExists(module, "relalg.limit");
    
    // TODO Phase 5: Implement actual lowering and verify SubOp operations
    // After lowering, should contain:
    // 1. subop.create_heap with maxRows=10
    // 2. subop.materialize to populate heap
    // 3. subop.scan to read results
    
    EXPECT_TRUE(limitOp);
    EXPECT_EQ(limitOp.getMaxRows(), 10);
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // verifyContainsSubOpType(module, "subop.create_heap");
    // verifyContainsSubOpType(module, "subop.materialize");
    // verifyContainsSubOpType(module, "subop.scan");
}

TEST_F(SortLimitLoweringTest, SortOperationLowering) {
    // Test that SortOp lowers to CreateSortedViewOp with comparison region
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto columnRef = createColumnRef("column1", builder->getI32Type());
    auto sortSpec = createSortSpec(columnRef, relalg::SortSpec::asc);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.sort");
    EXPECT_EQ(sortOp.getSortspecs().size(), 1);
    
    // TODO Phase 5: Implement actual lowering and verify SubOp operations
    // After lowering, should contain:
    // 1. subop.create (for buffer)
    // 2. subop.materialize to populate buffer  
    // 3. subop.create_sorted_view with comparison region
    // 4. subop.scan with sequential attribute
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // verifyContainsSubOpType(module, "subop.create_sorted_view");
    // verifyContainsSubOpType(module, "subop.materialize");
    // verifyContainsSubOpType(module, "subop.scan");
}

TEST_F(SortLimitLoweringTest, TopKOperationLowering) {
    // Test that TopKOp lowers to size-limited CreateHeapOp
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto columnRef = createColumnRef("score", builder->getF64Type());
    auto sortSpec = createSortSpec(columnRef, relalg::SortSpec::desc);
    
    auto topkOp = builder->create<relalg::TopKOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(5), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.topk");
    EXPECT_EQ(topkOp.getMaxRows(), 5);
    EXPECT_EQ(topkOp.getSortspecs().size(), 1);
    
    // TODO Phase 5: Implement actual lowering and verify SubOp operations
    // After lowering, should contain:
    // 1. subop.create_heap with max_elements=5 and comparison region
    // 2. subop.materialize to populate heap efficiently (O(n log k))
    // 3. subop.scan to read top-K results
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // verifyContainsSubOpType(module, "subop.create_heap");
    // verifyContainsSubOpType(module, "subop.materialize");
    // verifyContainsSubOpType(module, "subop.scan");
}

//===----------------------------------------------------------------------===//
// Multi-Column and Complex Sort Tests
//===----------------------------------------------------------------------===//

TEST_F(SortLimitLoweringTest, SortMultiColumnLowering) {
    // Test multi-column sort with mixed ASC/DESC
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto columnRef1 = createColumnRef("column1", builder->getI32Type());
    auto columnRef2 = createColumnRef("column2", builder->getF64Type());
    
    auto sortSpec1 = createSortSpec(columnRef1, relalg::SortSpec::desc);
    auto sortSpec2 = createSortSpec(columnRef2, relalg::SortSpec::asc);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec1, sortSpec2}));
    
    verifyRelAlgOpExists(module, "relalg.sort");
    EXPECT_EQ(sortOp.getSortspecs().size(), 2);
    
    // Verify sort specification ordering is preserved
    auto firstSpec = mlir::cast<relalg::SortSpecificationAttr>(
        sortOp.getSortspecs()[0]);
    auto secondSpec = mlir::cast<relalg::SortSpecificationAttr>(
        sortOp.getSortspecs()[1]);
    
    EXPECT_EQ(firstSpec.getSortSpec(), relalg::SortSpec::desc);
    EXPECT_EQ(secondSpec.getSortSpec(), relalg::SortSpec::asc);
    
    // TODO Phase 5: Verify comparison region handles multiple sort criteria
    // The spaceship comparison should check column1 DESC first, then column2 ASC
    
    // TODO Phase 5: Enable when lowering passes are implemented  
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // verifyContainsSubOpType(module, "subop.create_sorted_view");
}

TEST_F(SortLimitLoweringTest, TopKVsSortEfficiency) {
    // Test that TopK creates heap while Sort creates buffer + sorted view
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto columnRef = createColumnRef("column1", builder->getI32Type());
    auto sortSpec = createSortSpec(columnRef, relalg::SortSpec::asc);
    
    // Create TopK(10) - should use heap for efficiency
    auto topkOp = builder->create<relalg::TopKOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(10), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    // Create full Sort - should use buffer + sorted view
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.topk");
    verifyRelAlgOpExists(module, "relalg.sort");
    
    // TODO Phase 5: Verify different lowering strategies
    // TopK should create: CreateHeapOp (size-limited)
    // Sort should create: GenericCreateOp (buffer) + CreateSortedViewOp
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // 
    // auto heapCount = countSubOpType(module, "subop.create_heap");
    // auto sortedViewCount = countSubOpType(module, "subop.create_sorted_view");
    // auto bufferCount = countSubOpType(module, "subop.create");
    // 
    // EXPECT_GE(heapCount, 1); // TopK creates heap
    // EXPECT_GE(sortedViewCount, 1); // Sort creates sorted view
    // EXPECT_GE(bufferCount, 1); // Sort creates buffer
}

//===----------------------------------------------------------------------===//
// Edge Cases and Error Handling
//===----------------------------------------------------------------------===//

TEST_F(SortLimitLoweringTest, LimitZeroRowsLowering) {
    // Test edge case: LIMIT 0 should create minimal heap
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto limitOp = builder->create<relalg::LimitOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(0), 
        tableScan);
    
    verifyRelAlgOpExists(module, "relalg.limit");
    EXPECT_EQ(limitOp.getMaxRows(), 0);
    
    // TODO Phase 5: Verify heap with max_elements=0 is created
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // auto heapCount = countSubOpType(module, "subop.create_heap");
    // EXPECT_EQ(heapCount, 1);
}

TEST_F(SortLimitLoweringTest, EmptyInputGracefulHandling) {
    // Test that operations handle empty input relations gracefully
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto columnRef = createColumnRef("column1", builder->getI32Type());
    auto sortSpec = createSortSpec(columnRef, relalg::SortSpec::asc);
    
    // All operations should handle empty input without errors
    auto limitOp = builder->create<relalg::LimitOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(10), 
        tableScan);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    auto topkOp = builder->create<relalg::TopKOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        builder->getI32IntegerAttr(5), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.limit");
    verifyRelAlgOpExists(module, "relalg.sort");
    verifyRelAlgOpExists(module, "relalg.topk");
    
    // TODO Phase 5: Verify lowered operations handle empty input gracefully
    // All SubOp operations should handle empty materialization correctly
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
}

//===----------------------------------------------------------------------===//
// LingoDB Architecture Compliance Tests
//===----------------------------------------------------------------------===//

TEST_F(SortLimitLoweringTest, SpaceshipComparisonGeneration) {
    // Test that sort operations generate proper spaceship comparison regions
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    
    // Test different data types in sort
    auto intCol = createColumnRef("int_col", builder->getI32Type());
    auto floatCol = createColumnRef("float_col", builder->getF64Type());
    auto stringCol = createColumnRef("string_col", 
        builder->getType<db::StringType>());
    
    auto sortSpec1 = createSortSpec(intCol, relalg::SortSpec::asc);
    auto sortSpec2 = createSortSpec(floatCol, relalg::SortSpec::desc);
    auto sortSpec3 = createSortSpec(stringCol, relalg::SortSpec::asc);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec1, sortSpec2, sortSpec3}));
    
    verifyRelAlgOpExists(module, "relalg.sort");
    EXPECT_EQ(sortOp.getSortspecs().size(), 3);
    
    // TODO Phase 5: Verify spaceship comparison region handles:
    // 1. Type-specific comparisons (int, float, string)
    // 2. Null handling for each type (PostgreSQL-compatible: nulls last for ASC)
    // 3. Correct precedence ordering (first column has highest priority)
    // 4. Mixed ASC/DESC handling
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // 
    // // Verify the comparison region is generated correctly
    // module.walk([&](subop::CreateSortedViewOp op) {
    //     EXPECT_TRUE(op.getRegion().hasOneBbuilder->getUnknownLoc()k());
    //     // TODO: Verify comparison logic for mixed types and orderings
    // });
}

TEST_F(SortLimitLoweringTest, NullHandlingCompliance) {
    // Test PostgreSQL-compatible null handling in sorts
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto nullableCol = createColumnRef("nullable_col", 
        builder->getType<db::NullableType>(builder->getI32Type()));
    auto sortSpec = createSortSpec(nullableCol, relalg::SortSpec::asc);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.sort");
    
    // TODO Phase 5: Verify null handling in comparison regions
    // PostgreSQL standard: NULL values should sort last in ASC order
    //                     and first in DESC order
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // 
    // // Verify null handling logic in generated comparison regions
}

TEST_F(SortLimitLoweringTest, SortStabilityVerification) {
    // Test that sort implementation maintains stability for equal elements
    auto module = createTestModule();
    
    auto tableScan = createTestTableScan();
    auto primaryCol = createColumnRef("primary", builder->getI32Type());
    
    // Sort by primary only - secondary should maintain input order for ties
    auto sortSpec = createSortSpec(primaryCol, relalg::SortSpec::asc);
    
    auto sortOp = builder->create<relalg::SortOp>(builder->getUnknownLoc(), 
        tableScan.getType(), 
        tableScan, 
        builder->getArrayAttr({sortSpec}));
    
    verifyRelAlgOpExists(module, "relalg.sort");
    
    // TODO Phase 5: Verify spaceship comparison uses tuple position as tiebreaker
    // This ensures sort stability as required by LingoDB architecture
    
    // TODO Phase 5: Enable when lowering passes are implemented
    // ASSERT_TRUE(runLoweringPasses(module).succeeded());
    // 
    // // Verify comparison regions include tiebreaker logic for stability
}
