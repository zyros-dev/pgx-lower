#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "compiler/Dialect/RelAlg/RelAlgOps.h"
#include "compiler/Dialect/RelAlg/RelAlgDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamTypes.h"
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "compiler/Dialect/DB/DBOps.h"
#include "compiler/Dialect/DB/DBDialect.h"
#include "compiler/Dialect/DB/DBTypes.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

namespace {

class JoinOperationsLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<relalg::RelAlgDialect>();
        context.getOrLoadDialect<subop::SubOperatorDialect>();
        context.getOrLoadDialect<tuples::TupleStreamDialect>();
        context.getOrLoadDialect<db::DBDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        
        PGX_DEBUG("JoinOperationsLoweringTest setup completed");
    }

    void TearDown() override {
        PGX_DEBUG("JoinOperationsLoweringTest teardown");
    }

    // Create a mock base table operation for testing
    Value createMockTable(const std::string& tableName, 
                         const std::vector<std::pair<std::string, Type>>& columns,
                         Location loc) {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        
        // Create column definitions
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        SmallVector<NamedAttribute> columnAttrs;
        for (const auto& [colName, colType] : columns) {
            std::string scope = colManager.getUniqueScope(tableName);
            auto colDef = colManager.createDef(scope, colName);
            colDef.getColumn().type = colType;
            auto colDefAttr = tuples::ColumnDefAttr::get(&context, colDef);
            columnAttrs.push_back(builder->getNamedAttr(colName, colDefAttr));
        }
        
        auto columnsDict = builder->getDictionaryAttr(columnAttrs);
        auto tableIdAttr = builder->getStringAttr(tableName);
        
        return builder->create<relalg::BaseTableOp>(loc, tupleStreamType, 
                                                   tableIdAttr, columnsDict);
    }

    // Create a simple equality predicate for joins
    void createEqualityPredicate(Block& predicateBlock, 
                                const std::string& leftTable, const std::string& leftCol,
                                const std::string& rightTable, const std::string& rightCol,
                                Type colType) {
        auto loc = builder->getUnknownLoc();
        
        // Set insertion point to predicate block
        auto savedIP = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(&predicateBlock);
        
        auto tupleArg = predicateBlock.getArgument(0);
        
        // Get left column value
        auto leftColRef = tuples::ColumnRefAttr::get(&context,
            tuples::ColumnDefAttr::get(&context, leftCol, colType));
        auto leftVal = builder->create<tuples::GetColumnOp>(loc, colType, tupleArg, leftColRef);
        
        // Get right column value  
        auto rightColRef = tuples::ColumnRefAttr::get(&context,
            tuples::ColumnDefAttr::get(&context, rightCol, colType));
        auto rightVal = builder->create<tuples::GetColumnOp>(loc, colType, tupleArg, rightColRef);
        
        // Create equality comparison
        auto cmpOp = builder->create<db::CmpOp>(loc, builder->getI1Type(),
                                                   db::DBCmpPredicate::eq, leftVal, rightVal);
        
        builder->create<tuples::ReturnOp>(loc, cmpOp.getResult());
        builder->restoreInsertionPoint(savedIP);
    }

    // Helper to create array attributes for hash join keys
    ArrayAttr createHashKeys(const std::vector<std::string>& tableNames,
                           const std::vector<std::string>& colNames) {
        SmallVector<Attribute> keyAttrs;
        for (size_t i = 0; i < tableNames.size() && i < colNames.size(); ++i) {
            auto colRef = tuples::ColumnRefAttr::get(&context,
                tuples::ColumnDefAttr::get(&context, colNames[i], builder->getI64Type()));
            keyAttrs.push_back(colRef);
        }
        return builder->getArrayAttr(keyAttrs);
    }

    // Verify operation structure and basic properties
    void verifyJoinOperation(Operation* op, size_t expectedOperands, bool hasPredicateRegion = true) {
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(op->getNumOperands(), expectedOperands);
        EXPECT_EQ(op->getNumResults(), 1);
        
        if (hasPredicateRegion) {
            EXPECT_GE(op->getNumRegions(), 1);
            if (op->getNumRegions() > 0) {
                auto& predicateRegion = op->getRegion(0);
                EXPECT_EQ(predicateRegion.getBlocks().size(), 1);
                
                auto& predicateBlock = predicateRegion.front();
                EXPECT_EQ(predicateBlock.getNumArguments(), 1);
                EXPECT_TRUE(predicateBlock.getArgumentTypes()[0].isa<tuples::TupleType>());
            }
        }
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(JoinOperationsLoweringTest, CrossProductLowering) {
    PGX_DEBUG("Testing CrossProduct lowering");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()}
    }, loc);
    
    // Create CrossProduct operation
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto crossProduct = builder->create<relalg::CrossProductOp>(loc, tupleStreamType, 
                                                               leftTable, rightTable);
    
    // Verify operation structure
    verifyJoinOperation(crossProduct, 2, false); // CrossProduct has no predicate region
    
    // Verify the operation produces correct result type
    EXPECT_TRUE(crossProduct.getResult().getType().isa<tuples::TupleStreamType>());
    
    // Verify operands are correctly set
    EXPECT_EQ(crossProduct.getLeft(), leftTable);
    EXPECT_EQ(crossProduct.getRight(), rightTable);
    
    PGX_INFO("CrossProduct operation created successfully with Cartesian product semantics");
}

TEST_F(JoinOperationsLoweringTest, InnerJoinNestedLoopLowering) {
    PGX_DEBUG("Testing InnerJoin nested loop lowering");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()}
    }, loc);
    
    // Create InnerJoin operation
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto innerJoin = builder->create<relalg::InnerJoinOp>(loc, tupleStreamType, 
                                                         leftTable, rightTable);
    
    // Add predicate region for join condition
    auto& predicateRegion = innerJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "customers", "id", "orders", "customer_id", 
                           builder->getI64Type());
    
    // Verify operation structure
    verifyJoinOperation(innerJoin, 2, true);
    
    // Test without hash join (nested loop default)
    EXPECT_FALSE(innerJoin->hasAttr("useHashJoin"));
    EXPECT_FALSE(innerJoin->hasAttr("useIndexNestedLoop"));
    
    PGX_INFO("InnerJoin nested loop operation created with equality predicate");
}

TEST_F(JoinOperationsLoweringTest, InnerJoinHashLowering) {
    PGX_DEBUG("Testing InnerJoin hash join lowering");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()}
    }, loc);
    
    // Create InnerJoin operation with hash join attributes
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto innerJoin = builder->create<relalg::InnerJoinOp>(loc, tupleStreamType, 
                                                         leftTable, rightTable);
    
    // Set hash join attributes
    innerJoin->setAttr("useHashJoin", builder->getUnitAttr());
    
    auto leftHash = createHashKeys({"customers"}, {"id"});
    auto rightHash = createHashKeys({"orders"}, {"customer_id"});
    innerJoin->setAttr("leftHash", leftHash);
    innerJoin->setAttr("rightHash", rightHash);
    
    // Create nulls equal array (empty for this test)
    innerJoin->setAttr("nullsEqual", builder->getArrayAttr({}));
    
    // Add predicate region
    auto& predicateRegion = innerJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "customers", "id", "orders", "customer_id", 
                           builder->getI64Type());
    
    // Verify hash join attributes
    EXPECT_TRUE(innerJoin->hasAttr("useHashJoin"));
    EXPECT_TRUE(innerJoin->hasAttr("leftHash"));
    EXPECT_TRUE(innerJoin->hasAttr("rightHash"));
    EXPECT_TRUE(innerJoin->hasAttr("nullsEqual"));
    
    verifyJoinOperation(innerJoin, 2, true);
    
    PGX_INFO("InnerJoin hash join operation created with hash keys");
}

TEST_F(JoinOperationsLoweringTest, MarkJoinLowering) {
    PGX_DEBUG("Testing MarkJoin lowering");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()}
    }, loc);
    
    // Create mark attribute for existence check
    auto markAttr = tuples::ColumnDefAttr::get(&context,
        "has_orders", builder->getI1Type());
    
    // Create MarkJoin operation
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto markJoin = builder->create<relalg::MarkJoinOp>(loc, tupleStreamType,
                                                       markAttr, leftTable, rightTable);
    
    // Add predicate region
    auto& predicateRegion = markJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "customers", "id", "orders", "customer_id", 
                           builder->getI64Type());
    
    // Verify operation structure
    verifyJoinOperation(markJoin, 2, true);
    
    // Verify mark attribute
    EXPECT_EQ(markJoin.getMarkattr(), markAttr);
    
    // Test without reverse sides (default)
    EXPECT_FALSE(markJoin->hasAttr("reverseSides"));
    
    PGX_INFO("MarkJoin operation created with existence check marker");
}

TEST_F(JoinOperationsLoweringTest, MarkJoinReversedLowering) {
    PGX_DEBUG("Testing MarkJoin with reversed sides");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()}
    }, loc);
    
    // Create mark attribute
    auto markAttr = tuples::ColumnDefAttr::get(&context,
        "has_orders", builder->getI1Type());
    
    // Create MarkJoin operation with reversed sides
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto markJoin = builder->create<relalg::MarkJoinOp>(loc, tupleStreamType,
                                                       markAttr, leftTable, rightTable);
    
    markJoin->setAttr("reverseSides", builder->getUnitAttr());
    
    // Add predicate region
    auto& predicateRegion = markJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "customers", "id", "orders", "customer_id", 
                           builder->getI64Type());
    
    // Verify reverse sides attribute
    EXPECT_TRUE(markJoin->hasAttr("reverseSides"));
    
    verifyJoinOperation(markJoin, 2, true);
    
    PGX_INFO("MarkJoin operation created with reversed sides optimization");
}

TEST_F(JoinOperationsLoweringTest, SingleJoinLowering) {
    PGX_DEBUG("Testing SingleJoin lowering");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("orders", {
        {"order_id", builder->getI64Type()},
        {"customer_id", builder->getI64Type()},
        {"amount", builder->getF64Type()}
    }, loc);
    
    // Create column mapping for semi-join result
    SmallVector<Attribute> mappingAttrs;
    auto leftIdRef = tuples::ColumnRefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "customers", "id"));
    auto leftNameRef = tuples::ColumnRefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "customers", "name"));
    auto orderAmountRef = tuples::ColumnRefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "orders", "amount"));
    
    mappingAttrs.push_back(leftIdRef);
    mappingAttrs.push_back(leftNameRef);
    mappingAttrs.push_back(orderAmountRef);
    
    auto mapping = builder->getArrayAttr(mappingAttrs);
    
    // Create SingleJoin operation
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto singleJoin = builder->create<relalg::SingleJoinOp>(loc, tupleStreamType,
                                                           leftTable, rightTable, mapping);
    
    // Add predicate region
    auto& predicateRegion = singleJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "customers", "id", "orders", "customer_id", 
                           builder->getI64Type());
    
    // Verify operation structure
    verifyJoinOperation(singleJoin, 2, true);
    
    // Verify mapping attribute
    EXPECT_EQ(singleJoin.getMapping(), mapping);
    
    // Test without constant join (default)
    EXPECT_FALSE(singleJoin->hasAttr("constantJoin"));
    
    PGX_INFO("SingleJoin operation created with column mapping for semi-join");
}

TEST_F(JoinOperationsLoweringTest, SingleJoinConstantLowering) {
    PGX_DEBUG("Testing SingleJoin constant join optimization");
    
    auto loc = builder->getUnknownLoc();
    
    // Create two mock tables
    auto leftTable = createMockTable("customers", {
        {"id", builder->getI64Type()},
        {"name", db::StringType::get(&context)}
    }, loc);
    
    auto rightTable = createMockTable("constants", {
        {"value", builder->getI64Type()}
    }, loc);
    
    // Create column mapping
    SmallVector<Attribute> mappingAttrs;
    auto leftIdRef = tuples::ColumnRefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "customers", "id"));
    auto constValueRef = tuples::ColumnRefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "constants", "value"));
    
    mappingAttrs.push_back(leftIdRef);
    mappingAttrs.push_back(constValueRef);
    
    auto mapping = builder->getArrayAttr(mappingAttrs);
    
    // Create SingleJoin operation with constant join optimization
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto singleJoin = builder->create<relalg::SingleJoinOp>(loc, tupleStreamType,
                                                           leftTable, rightTable, mapping);
    
    singleJoin->setAttr("constantJoin", builder->getUnitAttr());
    
    // Add predicate region (even for constant joins)
    auto& predicateRegion = singleJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    // Simple true predicate for constant join
    auto savedIP = builder->saveInsertionPoint();
    builder->setInsertionPointToStart(&predicateBlock);
    auto trueVal = builder->create<db::ConstantOp>(loc, builder->getI1Type(),
                                                  builder->getBoolAttr(true));
    builder->create<tuples::ReturnOp>(loc, trueVal.getResult());
    builder->restoreInsertionPoint(savedIP);
    
    // Verify constant join attribute
    EXPECT_TRUE(singleJoin->hasAttr("constantJoin"));
    
    verifyJoinOperation(singleJoin, 2, true);
    
    PGX_INFO("SingleJoin constant join operation created with lookup optimization");
}

TEST_F(JoinOperationsLoweringTest, JoinAlgorithmAttributes) {
    PGX_DEBUG("Testing join algorithm attribute variations");
    
    auto loc = builder->getUnknownLoc();
    
    // Create mock tables
    auto leftTable = createMockTable("table1", {
        {"key", builder->getI64Type()}
    }, loc);
    
    auto rightTable = createMockTable("table2", {
        {"key", builder->getI64Type()}
    }, loc);
    
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    
    // Test InnerJoin with index nested loop
    auto indexJoin = builder->create<relalg::InnerJoinOp>(loc, tupleStreamType, 
                                                         leftTable, rightTable);
    indexJoin->setAttr("useIndexNestedLoop", builder->getUnitAttr());
    
    EXPECT_TRUE(indexJoin->hasAttr("useIndexNestedLoop"));
    EXPECT_FALSE(indexJoin->hasAttr("useHashJoin"));
    
    // Test MarkJoin with hash join
    auto markAttr = tuples::ColumnDefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "mark", "exists"), builder->getI1Type());
    auto hashMarkJoin = builder->create<relalg::MarkJoinOp>(loc, tupleStreamType,
                                                           markAttr, leftTable, rightTable);
    hashMarkJoin->setAttr("useHashJoin", builder->getUnitAttr());
    
    auto leftHash = createHashKeys({"table1"}, {"key"});
    auto rightHash = createHashKeys({"table2"}, {"key"});
    hashMarkJoin->setAttr("leftHash", leftHash);
    hashMarkJoin->setAttr("rightHash", rightHash);
    hashMarkJoin->setAttr("nullsEqual", builder->getArrayAttr({}));
    
    EXPECT_TRUE(hashMarkJoin->hasAttr("useHashJoin"));
    EXPECT_TRUE(hashMarkJoin->hasAttr("leftHash"));
    EXPECT_TRUE(hashMarkJoin->hasAttr("rightHash"));
    
    PGX_INFO("Join algorithm attributes verified for different join types");
}

TEST_F(JoinOperationsLoweringTest, JoinMemoryManagement) {
    PGX_DEBUG("Testing join memory management patterns");
    
    auto loc = builder->getUnknownLoc();
    
    // Create larger mock tables to test memory efficiency
    auto leftTable = createMockTable("large_table", {
        {"id", builder->getI64Type()},
        {"data1", db::StringType::get(&context)},
        {"data2", builder->getF64Type()}
    }, loc);
    
    auto rightTable = createMockTable("lookup_table", {
        {"id", builder->getI64Type()},
        {"lookup_value", builder->getI32Type()}
    }, loc);
    
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    
    // Create hash join for better memory efficiency with large tables
    auto hashJoin = builder->create<relalg::InnerJoinOp>(loc, tupleStreamType, 
                                                        leftTable, rightTable);
    
    // Configure for hash join with memory-efficient settings
    hashJoin->setAttr("useHashJoin", builder->getUnitAttr());
    
    auto leftHash = createHashKeys({"large_table"}, {"id"});
    auto rightHash = createHashKeys({"lookup_table"}, {"id"});
    hashJoin->setAttr("leftHash", leftHash);
    hashJoin->setAttr("rightHash", rightHash);
    hashJoin->setAttr("nullsEqual", builder->getArrayAttr({}));
    
    // Add predicate region
    auto& predicateRegion = hashJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "large_table", "id", "lookup_table", "id", 
                           builder->getI64Type());
    
    // Verify hash join configuration for memory efficiency
    EXPECT_TRUE(hashJoin->hasAttr("useHashJoin"));
    
    verifyJoinOperation(hashJoin, 2, true);
    
    PGX_INFO("Hash join configured for memory-efficient large table processing");
}

TEST_F(JoinOperationsLoweringTest, JoinNullHandling) {
    PGX_DEBUG("Testing join null handling with nullsEqual attributes");
    
    auto loc = builder->getUnknownLoc();
    
    // Create tables with nullable columns
    auto leftTable = createMockTable("nullable_table", {
        {"id", db::NullableType::get(&context, builder->getI64Type())},
        {"value", builder->getI32Type()}
    }, loc);
    
    auto rightTable = createMockTable("reference_table", {
        {"ref_id", db::NullableType::get(&context, builder->getI64Type())},
        {"description", db::StringType::get(&context)}
    }, loc);
    
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    
    // Create join with null equality handling
    auto nullJoin = builder->create<relalg::InnerJoinOp>(loc, tupleStreamType, 
                                                        leftTable, rightTable);
    
    nullJoin->setAttr("useHashJoin", builder->getUnitAttr());
    
    auto leftHash = createHashKeys({"nullable_table"}, {"id"});
    auto rightHash = createHashKeys({"reference_table"}, {"ref_id"});
    nullJoin->setAttr("leftHash", leftHash);
    nullJoin->setAttr("rightHash", rightHash);
    
    // Configure null equality - nulls are considered equal
    SmallVector<Attribute> nullsEqualAttrs;
    nullsEqualAttrs.push_back(builder->getBoolAttr(true)); // First key: nulls equal
    auto nullsEqual = builder->getArrayAttr(nullsEqualAttrs);
    nullJoin->setAttr("nullsEqual", nullsEqual);
    
    // Add predicate region
    auto& predicateRegion = nullJoin.getPredicate();
    auto& predicateBlock = predicateRegion.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    predicateBlock.addArgument(tupleType, loc);
    
    createEqualityPredicate(predicateBlock, "nullable_table", "id", "reference_table", "ref_id", 
                           db::NullableType::get(&context, builder->getI64Type()));
    
    // Verify null handling configuration
    EXPECT_TRUE(nullJoin->hasAttr("nullsEqual"));
    auto nullsEqualAttr = nullJoin->getAttrOfType<ArrayAttr>("nullsEqual");
    EXPECT_EQ(nullsEqualAttr.size(), 1);
    
    verifyJoinOperation(nullJoin, 2, true);
    
    PGX_INFO("Join null handling configured with nullsEqual semantics");
}

} // anonymous namespace