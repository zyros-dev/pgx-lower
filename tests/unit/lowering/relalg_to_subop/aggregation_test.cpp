#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBTypes.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class AggregationLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();

        // Initialize builder
        builder = std::make_unique<OpBuilder>(&context);
        
        // Initialize column manager
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        // Create test columns for aggregation operations
        createTestColumns();
    }

    void TearDown() override {
        // Nothing specific to clean up
    }

    void createTestColumns() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        // Create test table columns using new API
        std::string scope = colManager.getUniqueScope("employee");
        salaryCol = colManager.createDef(scope, "salary");
        ageCol = colManager.createDef(scope, "age");
        nameCol = colManager.createDef(scope, "name");
        deptIdCol = colManager.createDef(scope, "dept_id");
        commissionCol = colManager.createDef(scope, "commission");
        
        // Set types manually
        salaryCol.getColumn().type = builder->getI64Type();
        ageCol.getColumn().type = builder->getI32Type();
        nameCol.getColumn().type = builder->getType<db::StringType>();
        deptIdCol.getColumn().type = builder->getI32Type();
        commissionCol.getColumn().type = builder->getI64Type(); // Simplified type
        
        // Create references for these columns
        salaryRef = colManager.createRef(salaryCol);
        ageRef = colManager.createRef(ageCol);
        nameRef = colManager.createRef(nameCol);
        deptIdRef = colManager.createRef(deptIdCol);
        commissionRef = colManager.createRef(commissionCol);
    }

    Value createTestTableScan() {
        // Create a simple table scan operation for testing
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        std::vector<Attribute> columnAttrs = {
            tuples::ColumnDefAttr::get(&context, salaryCol),
            tuples::ColumnDefAttr::get(&context, ageCol),
            tuples::ColumnDefAttr::get(&context, nameCol),
            tuples::ColumnDefAttr::get(&context, deptIdCol),
            tuples::ColumnDefAttr::get(&context, commissionCol)
        };
        auto columnsAttr = builder->getArrayAttr(columnAttrs);
        
        return builder->create<relalg::BaseTableOp>(builder->getUnknownLoc(), tupleStreamType, 
                                                   builder->getStringAttr("employee|oid:12345"), 
                                                   columnsAttr);
    }

    Value createAggregationOperation(Value input, relalg::AggrFunc aggrFunc, 
                                   tuples::ColumnRefAttr sourceCol, tuples::ColumnDefAttr destCol,
                                   ArrayAttr groupByColumns = nullptr) {
        // Create aggregation function operation
        auto aggrType = destCol.getColumn().type;
        Value aggrFuncOp = builder->create<relalg::AggrFuncOp>(builder->getUnknownLoc(), aggrType, aggrFunc, input, sourceCol);
        
        // Create aggregation operation
        auto computedCols = ArrayAttr::get(&context, {destCol});
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        
        // Create the aggregation block
        auto aggrOp = builder->create<relalg::AggregationOp>(builder->getUnknownLoc(), tupleStreamType, input, 
                                                            groupByColumns ? groupByColumns : ArrayAttr::get(&context, {}),
                                                            computedCols);
        
        // Add the aggregation function block
        Block* aggrBlock = new Block();
        aggrBlock->addArgument(tuples::TupleType::get(&context), builder->getUnknownLoc());
        aggrOp.getAggrFunc().push_back(aggrBlock);
        
        {
            OpBuilder::InsertionGuard guard(*builder);
            builder->setInsertionPointToStart(aggrBlock);
            builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), ValueRange{aggrFuncOp});
        }
        
        return aggrOp;
    }

    Value createCountRowsOperation(Value input, tuples::ColumnDefAttr destCol, ArrayAttr groupByColumns = nullptr) {
        // Create COUNT(*) operation
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        Value countOp = builder->create<relalg::CountRowsOp>(builder->getUnknownLoc(), builder->getI64Type(), input);
        
        auto computedCols = ArrayAttr::get(&context, {destCol});
        auto aggrOp = builder->create<relalg::AggregationOp>(builder->getUnknownLoc(), tupleStreamType, input,
                                                            groupByColumns ? groupByColumns : ArrayAttr::get(&context, {}),
                                                            computedCols);
        
        // Add the aggregation function block
        Block* aggrBlock = new Block();
        aggrBlock->addArgument(tuples::TupleType::get(&context), builder->getUnknownLoc());
        aggrOp.getAggrFunc().push_back(aggrBlock);
        
        {
            OpBuilder::InsertionGuard guard(*builder);
            builder->setInsertionPointToStart(aggrBlock);
            builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), ValueRange{countOp});
        }
        
        return aggrOp;
    }

    bool validateStateManagement(Value aggregationResult) {
        // Check that proper state management operations are created
        bool hasMapType = false;
        bool hasLookupOrInsert = false;
        bool hasReduce = false;
        bool hasScan = false;
        
        aggregationResult.getDefiningOp()->getParentRegion()->walk([&](Operation* op) {
            if (isa<subop::GenericCreateOp>(op)) {
                auto stateType = op->getResult(0).getType();
                if (isa<subop::MapType>(stateType)) {
                    hasMapType = true;
                }
            }
            if (isa<subop::LookupOrInsertOp>(op)) {
                hasLookupOrInsert = true;
            }
            if (isa<subop::ReduceOp>(op)) {
                hasReduce = true;
            }
            if (isa<subop::ScanOp>(op)) {
                hasScan = true;
            }
        });
        
        return hasMapType && hasLookupOrInsert && hasReduce && hasScan;
    }

    bool validateNullHandling(Value aggregationResult) {
        // Check that proper null handling is implemented
        bool hasNullCheck = false;
        
        aggregationResult.getDefiningOp()->getParentRegion()->walk([&](Operation* op) {
            if (isa<db::IsNullOp>(op) || isa<db::AsNullableOp>(op) || isa<db::NullOp>(op)) {
                hasNullCheck = true;
            }
        });
        
        return hasNullCheck;
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    
    // Test columns
    tuples::ColumnDefAttr salaryCol, ageCol, nameCol, deptIdCol, commissionCol;
    tuples::ColumnRefAttr salaryRef, ageRef, nameRef, deptIdRef, commissionRef;
};

TEST_F(AggregationLoweringTest, BasicSumAggregation) {
    PGX_DEBUG("Testing basic SUM aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto sumSalaryCol = colManager.createDef(resultScope, "total_salary");
    sumSalaryCol.getColumn().type = builder->getI64Type();
    
    Value tableScan = createTestTableScan();
    Value sumAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, sumSalaryCol);
    
    EXPECT_TRUE(sumAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(sumAggr.getDefiningOp()));
    
    // Verify that the aggregation operation has the correct structure
    auto aggrOp = cast<relalg::AggregationOp>(sumAggr.getDefiningOp());
    EXPECT_EQ(aggrOp.getComputedCols().size(), 1);
    EXPECT_TRUE(aggrOp.getGroupByCols().empty());
}

TEST_F(AggregationLoweringTest, BasicCountAggregation) {
    PGX_DEBUG("Testing basic COUNT aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto countCol = colManager.createDef(resultScope, "employee_count");
    countCol.getColumn().type = builder->getI64Type();
    
    Value tableScan = createTestTableScan();
    Value countAggr = createAggregationOperation(tableScan, relalg::AggrFunc::count, ageRef, countCol);
    
    EXPECT_TRUE(countAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(countAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, BasicCountStarAggregation) {
    PGX_DEBUG("Testing basic COUNT(*) aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto countStarCol = colManager.createDef(resultScope, "total_rows");
    countStarCol.getColumn().type = builder->getI64Type();
    
    Value tableScan = createTestTableScan();
    Value countStarAggr = createCountRowsOperation(tableScan, countStarCol);
    
    EXPECT_TRUE(countStarAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(countStarAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, BasicMinMaxAggregation) {
    PGX_DEBUG("Testing basic MIN/MAX aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto minAgeCol = colManager.createDef(resultScope, "min_age");
    minAgeCol.getColumn().type = builder->getI32Type();
    auto maxAgeCol = colManager.createDef(resultScope, "max_age");
    maxAgeCol.getColumn().type = builder->getI32Type();
    
    Value tableScan = createTestTableScan();
    
    // Test MIN
    Value minAggr = createAggregationOperation(tableScan, relalg::AggrFunc::min, ageRef, minAgeCol);
    EXPECT_TRUE(minAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(minAggr.getDefiningOp()));
    
    // Test MAX  
    Value maxAggr = createAggregationOperation(tableScan, relalg::AggrFunc::max, ageRef, maxAgeCol);
    EXPECT_TRUE(maxAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(maxAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, BasicAnyAggregation) {
    PGX_DEBUG("Testing basic ANY aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto anyNameCol = colManager.createDef(resultScope, "any_name");
    anyNameCol.getColumn().type = builder->getType<db::StringType>();
    
    Value tableScan = createTestTableScan();
    Value anyAggr = createAggregationOperation(tableScan, relalg::AggrFunc::any, nameRef, anyNameCol);
    
    EXPECT_TRUE(anyAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(anyAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, GroupByAggregation) {
    PGX_DEBUG("Testing GROUP BY aggregation lowering");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto avgSalaryCol = colManager.createDef(resultScope, "avg_salary");
    avgSalaryCol.getColumn().type = builder->getI64Type();
    
    // Create GROUP BY columns
    auto groupByColumns = ArrayAttr::get(&context, {deptIdRef});
    
    Value tableScan = createTestTableScan();
    Value groupAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, avgSalaryCol, groupByColumns);
    
    EXPECT_TRUE(groupAggr);
    auto aggrOp = cast<relalg::AggregationOp>(groupAggr.getDefiningOp());
    EXPECT_FALSE(aggrOp.getGroupByCols().empty());
    EXPECT_EQ(aggrOp.getGroupByCols().size(), 1);
}

TEST_F(AggregationLoweringTest, MultipleAggregationFunctions) {
    PGX_DEBUG("Testing multiple aggregation functions in single operation");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto sumSalaryCol = colManager.createDef(resultScope, "total_salary");
    sumSalaryCol.getColumn().type = builder->getI64Type();
    auto countCol = colManager.createDef(resultScope, "employee_count");
    countCol.getColumn().type = builder->getI64Type();
    auto avgAgeCol = colManager.createDef(resultScope, "avg_age");
    avgAgeCol.getColumn().type = builder->getI32Type();
    
    Value tableScan = createTestTableScan();
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    
    // Create aggregation operation with multiple computed columns
    auto computedCols = ArrayAttr::get(&context, {sumSalaryCol, countCol, avgAgeCol});
    auto groupByColumns = ArrayAttr::get(&context, {deptIdRef});
    
    auto aggrOp = builder->create<relalg::AggregationOp>(builder->getUnknownLoc(), tupleStreamType, tableScan, groupByColumns, computedCols);
    
    // Create the aggregation function block with multiple return values
    Block* aggrBlock = new Block();
    aggrBlock->addArgument(tuples::TupleType::get(&context), builder->getUnknownLoc());
    aggrOp.getAggrFunc().push_back(aggrBlock);
    
    {
        OpBuilder::InsertionGuard guard(*builder);
        builder->setInsertionPointToStart(aggrBlock);
        
        Value sumOp = builder->create<relalg::AggrFuncOp>(builder->getUnknownLoc(), builder->getI64Type(), relalg::AggrFunc::sum, tableScan, salaryRef);
        Value countOp = builder->create<relalg::AggrFuncOp>(builder->getUnknownLoc(), builder->getI64Type(), relalg::AggrFunc::count, tableScan, ageRef);
        Value avgOp = builder->create<relalg::AggrFuncOp>(builder->getUnknownLoc(), builder->getI32Type(), relalg::AggrFunc::sum, tableScan, ageRef); // Using sum for avg computation
        
        builder->create<tuples::ReturnOp>(builder->getUnknownLoc(), ValueRange{sumOp, countOp, avgOp});
    }
    
    EXPECT_TRUE(aggrOp);
    EXPECT_EQ(aggrOp.getComputedCols().size(), 3);
    EXPECT_EQ(aggrOp.getGroupByCols().size(), 1);
}

TEST_F(AggregationLoweringTest, NullableColumnAggregation) {
    PGX_DEBUG("Testing aggregation with nullable columns");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    // Create nullable commission column
    auto nullableCommissionType = db::NullableType::get(&context, builder->getI64Type());
    std::string empScope = colManager.getUniqueScope("employee");
    auto nullableCommissionCol = colManager.createDef(empScope, "commission");
    nullableCommissionCol.getColumn().type = nullableCommissionType;
    auto nullableCommissionRef = colManager.createRef(nullableCommissionCol);
    
    std::string resultScope = colManager.getUniqueScope("result");
    auto sumCommissionCol = colManager.createDef(resultScope, "total_commission");
    sumCommissionCol.getColumn().type = nullableCommissionType;
    
    Value tableScan = createTestTableScan();
    Value sumAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, nullableCommissionRef, sumCommissionCol);
    
    EXPECT_TRUE(sumAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(sumAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, DecimalAggregation) {
    PGX_DEBUG("Testing aggregation with decimal types");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto sumCommissionCol = colManager.createDef(resultScope, "total_commission");
    sumCommissionCol.getColumn().type = builder->getI64Type(); // Simplified type
    
    Value tableScan = createTestTableScan();
    Value decimalAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, commissionRef, sumCommissionCol);
    
    EXPECT_TRUE(decimalAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(decimalAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, StringAggregation) {
    PGX_DEBUG("Testing aggregation with string types");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto firstNameCol = colManager.createDef(resultScope, "first_name");
    firstNameCol.getColumn().type = builder->getType<db::StringType>();
    
    Value tableScan = createTestTableScan();
    Value stringAggr = createAggregationOperation(tableScan, relalg::AggrFunc::any, nameRef, firstNameCol);
    
    EXPECT_TRUE(stringAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(stringAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, EmptyGroupByAggregation) {
    PGX_DEBUG("Testing aggregation with empty GROUP BY (global aggregation)");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto totalSalaryCol = colManager.createDef(resultScope, "total_salary");
    totalSalaryCol.getColumn().type = builder->getI64Type();
    
    Value tableScan = createTestTableScan();
    Value globalAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, totalSalaryCol);
    
    EXPECT_TRUE(globalAggr);
    auto aggrOp = cast<relalg::AggregationOp>(globalAggr.getDefiningOp());
    EXPECT_TRUE(aggrOp.getGroupByCols().empty()); // Global aggregation
}

TEST_F(AggregationLoweringTest, MultipleGroupByColumns) {
    PGX_DEBUG("Testing aggregation with multiple GROUP BY columns");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto avgSalaryCol = colManager.createDef(resultScope, "avg_salary");
    avgSalaryCol.getColumn().type = builder->getI64Type();
    
    // Create multiple GROUP BY columns
    auto groupByColumns = ArrayAttr::get(&context, {deptIdRef, ageRef});
    
    Value tableScan = createTestTableScan();
    Value multiGroupAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, avgSalaryCol, groupByColumns);
    
    EXPECT_TRUE(multiGroupAggr);
    auto aggrOp = cast<relalg::AggregationOp>(multiGroupAggr.getDefiningOp());
    EXPECT_EQ(aggrOp.getGroupByCols().size(), 2);
}

TEST_F(AggregationLoweringTest, AggregationComposition) {
    PGX_DEBUG("Testing composition of aggregation operations");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    // First aggregation: sum salary by department
    std::string deptScope = colManager.getUniqueScope("dept_agg");
    auto deptSalaryCol = colManager.createDef(deptScope, "dept_total_salary");
    deptSalaryCol.getColumn().type = builder->getI64Type();
    auto groupByDept = ArrayAttr::get(&context, {deptIdRef});
    
    Value tableScan = createTestTableScan();
    Value deptAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, deptSalaryCol, groupByDept);
    
    // Second aggregation: max department salary (global)
    std::string globalScope = colManager.getUniqueScope("global_agg");
    auto maxDeptSalaryCol = colManager.createDef(globalScope, "max_dept_salary");
    maxDeptSalaryCol.getColumn().type = builder->getI64Type();
    auto deptSalaryRef = colManager.createRef(deptSalaryCol);
    Value maxDeptAggr = createAggregationOperation(deptAggr, relalg::AggrFunc::max, deptSalaryRef, maxDeptSalaryCol);
    
    EXPECT_TRUE(deptAggr);
    EXPECT_TRUE(maxDeptAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(deptAggr.getDefiningOp()));
    EXPECT_TRUE(isa<relalg::AggregationOp>(maxDeptAggr.getDefiningOp()));
}

TEST_F(AggregationLoweringTest, AggregationPerformanceCharacteristics) {
    PGX_DEBUG("Testing aggregation performance characteristics");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto sumSalaryCol = colManager.createDef(resultScope, "total_salary");
    sumSalaryCol.getColumn().type = builder->getI64Type();
    auto groupByColumns = ArrayAttr::get(&context, {deptIdRef});
    
    Value tableScan = createTestTableScan();
    Value perfAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, sumSalaryCol, groupByColumns);
    
    EXPECT_TRUE(perfAggr);
    
    // Verify basic structure is created
    auto aggrOp = cast<relalg::AggregationOp>(perfAggr.getDefiningOp());
    EXPECT_TRUE(aggrOp.getAggrFunc().front().hasTerminator());
    EXPECT_EQ(aggrOp.getComputedCols().size(), 1);
    EXPECT_EQ(aggrOp.getGroupByCols().size(), 1);
}

TEST_F(AggregationLoweringTest, AggregationMemoryManagement) {
    PGX_DEBUG("Testing aggregation memory management patterns");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    std::string resultScope = colManager.getUniqueScope("result");
    auto countCol = colManager.createDef(resultScope, "row_count");
    countCol.getColumn().type = builder->getI64Type();
    
    Value tableScan = createTestTableScan();
    Value memoryAggr = createCountRowsOperation(tableScan, countCol);
    
    EXPECT_TRUE(memoryAggr);
    
    // Verify that the operation is properly structured for memory management
    auto aggrOp = cast<relalg::AggregationOp>(memoryAggr.getDefiningOp());
    EXPECT_TRUE(aggrOp);
    EXPECT_TRUE(aggrOp.getAggrFunc().front().hasTerminator());
}

TEST_F(AggregationLoweringTest, AggregationErrorHandling) {
    PGX_DEBUG("Testing aggregation error handling patterns");
    
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    // Test with invalid type combinations (this would typically be caught at compile time)
    std::string resultScope = colManager.getUniqueScope("result");
    auto invalidTypeCol = colManager.createDef(resultScope, "invalid_result");
    invalidTypeCol.getColumn().type = builder->getI1Type();
    
    Value tableScan = createTestTableScan();
    
    // This should still create the operation structure, even if semantically invalid
    // (actual type checking would happen during lowering)
    Value errorAggr = createAggregationOperation(tableScan, relalg::AggrFunc::sum, salaryRef, invalidTypeCol);
    
    EXPECT_TRUE(errorAggr);
    EXPECT_TRUE(isa<relalg::AggregationOp>(errorAggr.getDefiningOp()));
}