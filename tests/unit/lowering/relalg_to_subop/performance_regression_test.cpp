#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Diagnostics.h>

#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

// Performance measurement utilities
struct PerformanceMetrics {
    std::chrono::microseconds loweringTime{0};
    size_t irSizeBefore{0};
    size_t irSizeAfter{0};
    size_t memoryUsedBytes{0};
    size_t numOperationsLowered{0};
    size_t numSubOpOperationsCreated{0};
    bool loweringSucceeded{false};
    std::string errorMessage;
};

struct MemoryTracker {
    size_t peakMemory{0};
    size_t currentMemory{0};
    
    void allocate(size_t bytes) {
        currentMemory += bytes;
        peakMemory = std::max(peakMemory, currentMemory);
    }
    
    void deallocate(size_t bytes) {
        currentMemory = (currentMemory >= bytes) ? currentMemory - bytes : 0;
    }
    
    void reset() {
        peakMemory = 0;
        currentMemory = 0;
    }
};

class PerformanceRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("PerformanceRegressionTest: Setting up test environment");
        
        // Load all required dialects
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        
        // Initialize builder
        builder = std::make_unique<OpBuilder>(&context);
        
        // Reset memory tracker
        memoryTracker.reset();
        
        PGX_DEBUG("PerformanceRegressionTest: Setup completed");
    }

    void TearDown() override {
        PGX_DEBUG("PerformanceRegressionTest: Tearing down test environment");
        builder.reset();
    }

    // Create test table with specified number of columns
    relalg::BaseTableOp createTestTable(const std::string& tableName, size_t numColumns) {
        auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        std::vector<NamedAttribute> columns;
        
        for (size_t i = 0; i < numColumns; ++i) {
            std::string columnName = "col" + std::to_string(i);
            std::string scopeName = columnManager.getUniqueScope("table");
            auto columnDef = columnManager.createDef(scopeName, columnName);
            columnDef.getColumn().type = builder->getI32Type(); // Use I32 for simplicity
            columns.push_back(builder->getNamedAttr(columnName, tuples::ColumnDefAttr::get(&context, columnDef)));
        }
        
        auto columnsAttr = builder->getDictionaryAttr(columns);
        return builder->create<relalg::BaseTableOp>(
            builder->getUnknownLoc(), 
            tuples::TupleStreamType::get(&context),
            builder->getStringAttr(tableName + "|oid:12345"),
            columnsAttr
        );
    }
    
    // Create complex join tree for scalability testing
    Operation* createComplexJoinTree(size_t depth, size_t numTablesPerLevel) {
        std::vector<Operation*> currentLevel;
        
        // Create base tables for the bottom level
        for (size_t i = 0; i < numTablesPerLevel; ++i) {
            std::string tableName = "table_level0_" + std::to_string(i);
            auto table = createTestTable(tableName, 5); // 5 columns per table
            currentLevel.push_back(table.getOperation());
        }
        
        // Build join tree level by level
        for (size_t level = 1; level < depth; ++level) {
            std::vector<Operation*> nextLevel;
            
            for (size_t i = 0; i < currentLevel.size(); i += 2) {
                if (i + 1 < currentLevel.size()) {
                    // Create join between two operations
                    auto leftOp = currentLevel[i];
                    auto rightOp = currentLevel[i + 1];
                    
                    // Create simple join predicate (true for simplicity)
                    auto predicateBlock = std::make_unique<Block>();
                    OpBuilder predicateBuilder(&context);
                    predicateBuilder.setInsertionPointToEnd(predicateBlock.get());
                    auto trueValue = predicateBuilder.create<arith::ConstantOp>(
                        builder->getUnknownLoc(), 
                        predicateBuilder.getI1Type(), 
                        predicateBuilder.getBoolAttr(true)
                    );
                    predicateBuilder.create<tuples::ReturnOp>(builder->getUnknownLoc(), trueValue.getResult());
                    
                    auto joinOp = builder->create<relalg::InnerJoinOp>(
                        builder->getUnknownLoc(),
                        tuples::TupleStreamType::get(&context),
                        leftOp->getResult(0),
                        rightOp->getResult(0)
                    );
                    joinOp.getPredicate().push_back(predicateBlock.release());
                    
                    nextLevel.push_back(joinOp.getOperation());
                } else {
                    // Odd number of operations, carry forward
                    nextLevel.push_back(currentLevel[i]);
                }
            }
            
            currentLevel = std::move(nextLevel);
        }
        
        return currentLevel.empty() ? nullptr : currentLevel[0];
    }
    
    // Create complex selection with multiple predicates
    relalg::SelectionOp createComplexSelection(Operation* inputOp, size_t numPredicates) {
        // Create selection with complex AND predicate
        auto selectionOp = builder->create<relalg::SelectionOp>(
            builder->getUnknownLoc(),
            tuples::TupleStreamType::get(&context),
            inputOp->getResult(0)
        );
        
        // Create predicate block with multiple AND conditions
        auto predicateBlock = std::make_unique<Block>();
        OpBuilder predicateBuilder(&context);
        predicateBuilder.setInsertionPointToEnd(predicateBlock.get());
        
        mlir::Value result = predicateBuilder.create<arith::ConstantOp>(
            builder->getUnknownLoc(), 
            predicateBuilder.getI1Type(), 
            predicateBuilder.getBoolAttr(true)
        ).getResult();
        
        // Add multiple predicates (simulated with constants for testing)
        for (size_t i = 0; i < numPredicates; ++i) {
            auto predicate = predicateBuilder.create<arith::ConstantOp>(
                builder->getUnknownLoc(),
                predicateBuilder.getI1Type(),
                predicateBuilder.getBoolAttr(true)
            );
            result = predicateBuilder.create<arith::AndIOp>(
                builder->getUnknownLoc(),
                predicateBuilder.getI1Type(),
                result,
                predicate.getResult()
            ).getResult();
        }
        
        predicateBuilder.create<tuples::ReturnOp>(builder->getUnknownLoc(), result);
        selectionOp.getPredicate().push_back(predicateBlock.release());
        
        return selectionOp;
    }
    
    // Measure lowering performance
    PerformanceMetrics measureLoweringTime(Operation* operation) {
        PerformanceMetrics metrics;
        
        // Create module containing the operation
        auto module = ModuleOp::create(builder->getUnknownLoc());
        auto funcOp = func::FuncOp::create(
            builder->getUnknownLoc(), 
            "test_func", 
            builder->getFunctionType({}, {operation->getResult(0).getType()})
        );
        
        auto& entryBlock = funcOp.getBody().emplaceBlock();
        OpBuilder funcBuilder(&context);
        funcBuilder.setInsertionPointToEnd(&entryBlock);
        
        // Clone the operation into the function
        auto clonedOp = funcBuilder.clone(*operation);
        funcBuilder.create<func::ReturnOp>(builder->getUnknownLoc(), clonedOp->getResult(0));
        
        module.push_back(funcOp);
        
        // Count operations before lowering
        module.walk([&](Operation* op) {
            if (isa<relalg::BaseTableOp, relalg::SelectionOp, relalg::InnerJoinOp, 
                    relalg::AggregationOp, relalg::ProjectionOp, relalg::SortOp>(op)) {
                metrics.numOperationsLowered++;
            }
        });
        
        // Measure IR size before
        std::string irBefore;
        llvm::raw_string_ostream streamBefore(irBefore);
        module.print(streamBefore);
        metrics.irSizeBefore = irBefore.size();
        
        // Create pass manager and add lowering pass
        PassManager pm(&context);
        pm.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
        
        // Measure lowering time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        auto result = pm.run(module);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        metrics.loweringTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        
        metrics.loweringSucceeded = succeeded(result);
        
        if (metrics.loweringSucceeded) {
            // Measure IR size after
            std::string irAfter;
            llvm::raw_string_ostream streamAfter(irAfter);
            module.print(streamAfter);
            metrics.irSizeAfter = irAfter.size();
            
            // Count SubOp operations created
            module.walk([&](Operation* op) {
                if (isa<subop::ScanRefsOp, subop::GatherOp, subop::FilterOp, 
                        subop::MapOp, subop::GenericCreateOp, subop::InsertOp>(op)) {
                    metrics.numSubOpOperationsCreated++;
                }
            });
        }
        
        metrics.memoryUsedBytes = memoryTracker.peakMemory;
        
        return metrics;
    }
    
    // Validate performance regression against baseline
    bool checkPerformanceRegression(const PerformanceMetrics& current, const PerformanceMetrics& baseline, double threshold = 1.5) {
        if (!baseline.loweringSucceeded || !current.loweringSucceeded) {
            return false;
        }
        
        // Check if current performance is more than threshold times slower than baseline
        double timeRatio = static_cast<double>(current.loweringTime.count()) / baseline.loweringTime.count();
        double memoryRatio = static_cast<double>(current.memoryUsedBytes) / baseline.memoryUsedBytes;
        
        return timeRatio < threshold && memoryRatio < threshold;
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    MemoryTracker memoryTracker;
};

// Performance baseline tests
TEST_F(PerformanceRegressionTest, LoweringPerformanceBaseline) {
    PGX_DEBUG("PerformanceRegressionTest: Testing lowering performance baseline");
    
    // Test simple BaseTable lowering
    auto baseTable = createTestTable("test_table", 5);
    auto metrics = measureLoweringTime(baseTable.getOperation());
    
    EXPECT_TRUE(metrics.loweringSucceeded) << "BaseTable lowering should succeed";
    EXPECT_GT(metrics.numSubOpOperationsCreated, 0) << "Should create SubOp operations";
    EXPECT_LT(metrics.loweringTime.count(), 10000) << "Lowering should complete within 10ms";
    
    PGX_INFO("BaseTable lowering metrics:");
    PGX_INFO("  Time: " + std::to_string(metrics.loweringTime.count()) + " microseconds");
    PGX_INFO("  IR size before: " + std::to_string(metrics.irSizeBefore) + " chars");
    PGX_INFO("  IR size after: " + std::to_string(metrics.irSizeAfter) + " chars");
    PGX_INFO("  SubOp operations created: " + std::to_string(metrics.numSubOpOperationsCreated));
}

TEST_F(PerformanceRegressionTest, SelectionLoweringPerformance) {
    PGX_DEBUG("PerformanceRegressionTest: Testing selection lowering performance");
    
    auto baseTable = createTestTable("test_table", 10);
    auto selection = createComplexSelection(baseTable.getOperation(), 5);
    auto metrics = measureLoweringTime(selection.getOperation());
    
    EXPECT_TRUE(metrics.loweringSucceeded) << "Selection lowering should succeed";
    EXPECT_LT(metrics.loweringTime.count(), 15000) << "Selection lowering should complete within 15ms";
    
    PGX_INFO("Selection lowering metrics:");
    PGX_INFO("  Time: " + std::to_string(metrics.loweringTime.count()) + " microseconds");
    PGX_INFO("  SubOp operations created: " + std::to_string(metrics.numSubOpOperationsCreated));
}

TEST_F(PerformanceRegressionTest, JoinLoweringPerformance) {
    PGX_DEBUG("PerformanceRegressionTest: Testing join lowering performance");
    
    auto leftTable = createTestTable("left_table", 5);
    auto rightTable = createTestTable("right_table", 5);
    
    // Create simple join predicate
    auto predicateBlock = std::make_unique<Block>();
    OpBuilder predicateBuilder(&context);
    predicateBuilder.setInsertionPointToEnd(predicateBlock.get());
    auto trueValue = predicateBuilder.create<arith::ConstantOp>(
        builder->getUnknownLoc(), 
        predicateBuilder.getI1Type(), 
        predicateBuilder.getBoolAttr(true)
    );
    predicateBuilder.create<tuples::ReturnOp>(builder->getUnknownLoc(), trueValue.getResult());
    
    auto joinOp = builder->create<relalg::InnerJoinOp>(
        builder->getUnknownLoc(),
        tuples::TupleStreamType::get(&context),
        leftTable.getResult(),
        rightTable.getResult()
    );
    joinOp.getPredicate().push_back(predicateBlock.release());
    
    auto metrics = measureLoweringTime(joinOp.getOperation());
    
    EXPECT_TRUE(metrics.loweringSucceeded) << "Join lowering should succeed";
    EXPECT_LT(metrics.loweringTime.count(), 20000) << "Join lowering should complete within 20ms";
    
    PGX_INFO("Join lowering metrics:");
    PGX_INFO("  Time: " + std::to_string(metrics.loweringTime.count()) + " microseconds");
    PGX_INFO("  SubOp operations created: " + std::to_string(metrics.numSubOpOperationsCreated));
}

// Memory usage validation tests
TEST_F(PerformanceRegressionTest, MemoryUsageValidation) {
    PGX_DEBUG("PerformanceRegressionTest: Testing memory usage during lowering");
    
    // Test with progressively larger operations
    std::vector<size_t> tableSizes = {5, 10, 20, 50};
    std::vector<PerformanceMetrics> memoryMetrics;
    
    for (size_t tableSize : tableSizes) {
        memoryTracker.reset();
        auto baseTable = createTestTable("test_table_" + std::to_string(tableSize), tableSize);
        auto metrics = measureLoweringTime(baseTable.getOperation());
        memoryMetrics.push_back(metrics);
        
        EXPECT_TRUE(metrics.loweringSucceeded) << "Lowering should succeed for table size " << tableSize;
        
        PGX_INFO("Memory usage for table with " + std::to_string(tableSize) + " columns: " + 
                std::to_string(metrics.memoryUsedBytes) + " bytes");
    }
    
    // Check that memory usage scales reasonably (should be roughly linear with table size)
    if (memoryMetrics.size() >= 2) {
        auto ratio1 = static_cast<double>(memoryMetrics[1].memoryUsedBytes) / memoryMetrics[0].memoryUsedBytes;
        auto ratio2 = static_cast<double>(memoryMetrics.back().memoryUsedBytes) / memoryMetrics[0].memoryUsedBytes;
        
        // Memory should scale but not exponentially
        EXPECT_LT(ratio2, 20.0) << "Memory usage should not scale exponentially with table size";
    }
}

// Scalability tests
TEST_F(PerformanceRegressionTest, ScalabilityWithComplexJoins) {
    PGX_DEBUG("PerformanceRegressionTest: Testing scalability with complex join trees");
    
    // Test join trees of increasing depth
    std::vector<size_t> depths = {2, 3, 4};
    std::vector<PerformanceMetrics> joinMetrics;
    
    for (size_t depth : depths) {
        auto complexJoin = createComplexJoinTree(depth, 2); // 2 tables per level
        if (complexJoin) {
            auto metrics = measureLoweringTime(complexJoin);
            joinMetrics.push_back(metrics);
            
            EXPECT_TRUE(metrics.loweringSucceeded) << "Complex join lowering should succeed for depth " << depth;
            
            PGX_INFO("Complex join metrics for depth " + std::to_string(depth) + ":");
            PGX_INFO("  Time: " + std::to_string(metrics.loweringTime.count()) + " microseconds");
            PGX_INFO("  Operations lowered: " + std::to_string(metrics.numOperationsLowered));
            PGX_INFO("  SubOp operations created: " + std::to_string(metrics.numSubOpOperationsCreated));
        }
    }
    
    // Verify that performance scales reasonably with complexity
    if (joinMetrics.size() >= 2) {
        // Time should scale but not exponentially
        auto timeRatio = static_cast<double>(joinMetrics.back().loweringTime.count()) / 
                        joinMetrics[0].loweringTime.count();
        EXPECT_LT(timeRatio, 10.0) << "Lowering time should not scale exponentially with join complexity";
    }
}

TEST_F(PerformanceRegressionTest, ScalabilityWithWideProjections) {
    PGX_DEBUG("PerformanceRegressionTest: Testing scalability with wide table projections");
    
    // Test tables with increasing column counts
    std::vector<size_t> columnCounts = {10, 25, 50, 100};
    std::vector<PerformanceMetrics> projectionMetrics;
    
    for (size_t columnCount : columnCounts) {
        auto baseTable = createTestTable("wide_table_" + std::to_string(columnCount), columnCount);
        auto metrics = measureLoweringTime(baseTable.getOperation());
        projectionMetrics.push_back(metrics);
        
        EXPECT_TRUE(metrics.loweringSucceeded) << "Wide table lowering should succeed for " << columnCount << " columns";
        
        PGX_INFO("Wide table metrics for " + std::to_string(columnCount) + " columns:");
        PGX_INFO("  Time: " + std::to_string(metrics.loweringTime.count()) + " microseconds");
        PGX_INFO("  IR size ratio: " + std::to_string(static_cast<double>(metrics.irSizeAfter) / metrics.irSizeBefore));
    }
    
    // Check that IR size growth is reasonable
    if (projectionMetrics.size() >= 2) {
        auto sizeRatio = static_cast<double>(projectionMetrics.back().irSizeAfter) / 
                        projectionMetrics[0].irSizeAfter;
        auto columnRatio = static_cast<double>(columnCounts.back()) / columnCounts[0];
        
        // IR size should scale roughly linearly with column count
        EXPECT_LT(sizeRatio, columnRatio * 2.0) << "IR size should scale reasonably with column count";
    }
}

// Regression detection tests
TEST_F(PerformanceRegressionTest, RegressionDetectionFramework) {
    PGX_DEBUG("PerformanceRegressionTest: Testing regression detection framework");
    
    // Create baseline performance profile
    auto baseTable = createTestTable("baseline_table", 10);
    auto baselineMetrics = measureLoweringTime(baseTable.getOperation());
    
    EXPECT_TRUE(baselineMetrics.loweringSucceeded) << "Baseline measurement should succeed";
    
    // Simulate current performance (should be similar to baseline)
    auto currentTable = createTestTable("current_table", 10);
    auto currentMetrics = measureLoweringTime(currentTable.getOperation());
    
    EXPECT_TRUE(currentMetrics.loweringSucceeded) << "Current measurement should succeed";
    
    // Verify regression detection works
    bool noRegression = checkPerformanceRegression(currentMetrics, baselineMetrics, 1.5);
    EXPECT_TRUE(noRegression) << "Should not detect regression for similar performance";
    
    PGX_INFO("Baseline vs Current Performance:");
    PGX_INFO("  Baseline time: " + std::to_string(baselineMetrics.loweringTime.count()) + " μs");
    PGX_INFO("  Current time: " + std::to_string(currentMetrics.loweringTime.count()) + " μs");
    PGX_INFO("  Performance ratio: " + std::to_string(
        static_cast<double>(currentMetrics.loweringTime.count()) / baselineMetrics.loweringTime.count()));
}

TEST_F(PerformanceRegressionTest, CompilationTimeRegression) {
    PGX_DEBUG("PerformanceRegressionTest: Testing compilation time regression");
    
    // Test compilation time for different operation types
    std::vector<std::string> operationTypes = {"BaseTable", "Selection", "Join"};
    std::unordered_map<std::string, PerformanceMetrics> compilationTimes;
    
    // BaseTable
    auto baseTable = createTestTable("perf_table", 15);
    compilationTimes["BaseTable"] = measureLoweringTime(baseTable.getOperation());
    
    // Selection
    auto selection = createComplexSelection(baseTable.getOperation(), 3);
    compilationTimes["Selection"] = measureLoweringTime(selection.getOperation());
    
    // Join
    auto leftTable = createTestTable("left_perf", 8);
    auto rightTable = createTestTable("right_perf", 8);
    
    auto predicateBlock = std::make_unique<Block>();
    OpBuilder predicateBuilder(&context);
    predicateBuilder.setInsertionPointToEnd(predicateBlock.get());
    auto trueValue = predicateBuilder.create<arith::ConstantOp>(
        builder->getUnknownLoc(), 
        predicateBuilder.getI1Type(), 
        predicateBuilder.getBoolAttr(true)
    );
    predicateBuilder.create<tuples::ReturnOp>(builder->getUnknownLoc(), trueValue.getResult());
    
    auto joinOp = builder->create<relalg::InnerJoinOp>(
        builder->getUnknownLoc(),
        tuples::TupleStreamType::get(&context),
        leftTable.getResult(),
        rightTable.getResult()
    );
    joinOp.getPredicate().push_back(predicateBlock.release());
    
    compilationTimes["Join"] = measureLoweringTime(joinOp.getOperation());
    
    // Verify all operations compile within reasonable time bounds
    for (const auto& [opType, metrics] : compilationTimes) {
        EXPECT_TRUE(metrics.loweringSucceeded) << opType << " lowering should succeed";
        EXPECT_LT(metrics.loweringTime.count(), 50000) << opType << " should compile within 50ms";
        
        PGX_INFO(opType + " compilation time: " + std::to_string(metrics.loweringTime.count()) + " μs");
    }
}

TEST_F(PerformanceRegressionTest, OutputQualityConsistency) {
    PGX_DEBUG("PerformanceRegressionTest: Testing output quality consistency");
    
    // Create identical operations and verify consistent lowering results
    std::vector<PerformanceMetrics> consistencyMetrics;
    
    for (int iteration = 0; iteration < 3; ++iteration) {
        auto baseTable = createTestTable("consistency_table", 12);
        auto metrics = measureLoweringTime(baseTable.getOperation());
        consistencyMetrics.push_back(metrics);
        
        EXPECT_TRUE(metrics.loweringSucceeded) << "Iteration " << iteration << " should succeed";
    }
    
    // Verify consistent results across iterations
    if (consistencyMetrics.size() >= 2) {
        auto& first = consistencyMetrics[0];
        
        for (size_t i = 1; i < consistencyMetrics.size(); ++i) {
            auto& current = consistencyMetrics[i];
            
            // Should create same number of SubOp operations
            EXPECT_EQ(current.numSubOpOperationsCreated, first.numSubOpOperationsCreated)
                << "Iteration " << i << " should create consistent number of SubOp operations";
            
            // IR size should be identical (deterministic lowering)
            EXPECT_EQ(current.irSizeAfter, first.irSizeAfter)
                << "Iteration " << i << " should produce identical IR size";
        }
    }
    
    PGX_INFO("Output quality consistency verified across " + 
            std::to_string(consistencyMetrics.size()) + " iterations");
}

// Comprehensive performance profiling test
TEST_F(PerformanceRegressionTest, ComprehensivePerformanceProfile) {
    PGX_DEBUG("PerformanceRegressionTest: Creating comprehensive performance profile");
    
    struct ProfileEntry {
        std::string testName;
        size_t complexity;
        PerformanceMetrics metrics;
    };
    
    std::vector<ProfileEntry> profile;
    
    // Single table operations
    for (size_t cols : {5, 10, 20}) {
        auto table = createTestTable("profile_table_" + std::to_string(cols), cols);
        auto metrics = measureLoweringTime(table.getOperation());
        profile.push_back({"BaseTable_" + std::to_string(cols), cols, metrics});
    }
    
    // Selection operations
    for (size_t predicates : {1, 3, 5}) {
        auto table = createTestTable("selection_table", 10);
        auto selection = createComplexSelection(table.getOperation(), predicates);
        auto metrics = measureLoweringTime(selection.getOperation());
        profile.push_back({"Selection_" + std::to_string(predicates), predicates, metrics});
    }
    
    // Join operations  
    for (size_t depth : {2, 3}) {
        auto joinTree = createComplexJoinTree(depth, 2);
        if (joinTree) {
            auto metrics = measureLoweringTime(joinTree);
            profile.push_back({"JoinTree_depth" + std::to_string(depth), depth, metrics});
        }
    }
    
    // Generate performance report
    PGX_INFO("=== Comprehensive Performance Profile ===");
    for (const auto& entry : profile) {
        EXPECT_TRUE(entry.metrics.loweringSucceeded) << entry.testName << " should succeed";
        
        PGX_INFO(entry.testName + ":");
        PGX_INFO("  Complexity: " + std::to_string(entry.complexity));
        PGX_INFO("  Time: " + std::to_string(entry.metrics.loweringTime.count()) + " μs");
        PGX_INFO("  Memory: " + std::to_string(entry.metrics.memoryUsedBytes) + " bytes");
        PGX_INFO("  Operations lowered: " + std::to_string(entry.metrics.numOperationsLowered));
        PGX_INFO("  SubOp ops created: " + std::to_string(entry.metrics.numSubOpOperationsCreated));
        PGX_INFO("  IR expansion: " + std::to_string(
            static_cast<double>(entry.metrics.irSizeAfter) / entry.metrics.irSizeBefore) + "x");
    }
    
    // Verify overall performance characteristics
    auto totalOperations = std::accumulate(profile.begin(), profile.end(), 0UL,
        [](size_t sum, const ProfileEntry& entry) { 
            return sum + entry.metrics.numOperationsLowered; 
        });
    
    auto totalTime = std::accumulate(profile.begin(), profile.end(), 0UL,
        [](size_t sum, const ProfileEntry& entry) { 
            return sum + entry.metrics.loweringTime.count(); 
        });
    
    EXPECT_GT(totalOperations, 0) << "Should have lowered operations across all tests";
    EXPECT_LT(totalTime, 500000) << "Total lowering time should be under 500ms";
    
    PGX_INFO("=== Profile Summary ===");
    PGX_INFO("Total operations lowered: " + std::to_string(totalOperations));
    PGX_INFO("Total lowering time: " + std::to_string(totalTime) + " μs");
    PGX_INFO("Average time per operation: " + std::to_string(totalTime / std::max(1UL, totalOperations)) + " μs");
}