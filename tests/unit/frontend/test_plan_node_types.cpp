#include <gtest/gtest.h>
#include "frontend/SQL/postgresql_ast_translator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "execution/logging.h"

// Mock PostgreSQL structures for testing
extern "C" {
    // Basic types
    typedef int16_t int16;
    typedef int16 AttrNumber;
    typedef unsigned int Oid;
    
    // Mock plan nodes for unit testing
    struct Plan {
        int type;
        Plan* lefttree;
        Plan* righttree;
        List* targetlist;
    };
    
    struct SeqScan {
        Plan plan;
        struct {
            int scanrelid;
        } scan;
    };
    
    struct Agg {
        Plan plan;
        int aggstrategy;
        int numCols;
        AttrNumber* grpColIdx;
        Oid* grpOperators;
        Oid* grpCollations;
    };
    
    struct Sort {
        Plan plan;
        int numCols;
        AttrNumber* sortColIdx;
        Oid* sortOperators;
        Oid* collations;
        bool* nullsFirst;
    };
    
    struct Limit {
        Plan plan;
        Node* limitCount;
        Node* limitOffset;
    };
    
    struct Gather {
        Plan plan;
        int num_workers;
        int rescan_param;
        bool single_copy;
        bool invisible;
    };
    
    struct PlannedStmt {
        int commandType;
        Plan* planTree;
        List* rtable;
    };
    
    struct List {
        void* head;
    };
    
    struct Node {
        int type;
    };
    
    struct RangeTblEntry {
        unsigned int relid;
    };
    
    struct Const {
        Node node;
        Oid consttype;
        int32_t consttypmod;
        Oid constcollid;
        int constlen;
        unsigned long constvalue;
        bool constisnull;
        bool constbyval;
        int location;
    };
    
    struct Param {
        Node node;
        int paramkind;
        int paramid;
        Oid paramtype;
        int32_t paramtypmod;
        Oid paramcollid;
        int location;
    };
    
    // Plan node type constants
    #define T_SeqScan 335
    #define T_Agg 361
    #define T_Sort 358
    #define T_Limit 369
    #define T_Gather 364
    #define T_Const 400
    #define T_Param 401
    
    // Aggregate strategy constants
    #define AGG_PLAIN 0
    #define AGG_SORTED 1
    #define AGG_HASHED 2
    #define AGG_MIXED 3
}

class PlanNodeTranslationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize MLIR context with required dialects
        context = std::make_unique<mlir::MLIRContext>();
        context->loadDialect<mlir::relalg::RelAlgDialect>();
        context->loadDialect<mlir::dsa::DSADialect>();
        context->loadDialect<mlir::util::UtilDialect>();
        context->loadDialect<mlir::db::DBDialect>();
        
        // Create translator
        translator = postgresql_ast::createPostgreSQLASTTranslator(*context);
    }
    
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<postgresql_ast::PostgreSQLASTTranslator> translator;
};

TEST_F(PlanNodeTranslationTest, TranslatesSeqScanNode) {
    PGX_INFO("Testing SeqScan node translation");
    
    // Create mock SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&seqScan);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "SeqScan translation should produce a module";
    PGX_INFO("SeqScan node translated successfully");
}

TEST_F(PlanNodeTranslationTest, TranslatesAggNode) {
    PGX_INFO("Testing Agg node translation");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Agg node with SeqScan as child
    Agg agg{};
    agg.plan.type = T_Agg;
    agg.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    agg.plan.righttree = nullptr;
    agg.aggstrategy = AGG_PLAIN;
    agg.numCols = 1;
    
    // Setup group by columns
    AttrNumber grpCols[] = {1};
    agg.grpColIdx = grpCols;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&agg);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Agg translation should produce a module";
    
    // Verify the module contains expected operations
    bool hasAggOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.aggregation") {
            hasAggOp = true;
            PGX_INFO("Found AggregationOp in generated MLIR");
        }
    });
    
    EXPECT_TRUE(hasAggOp) << "Module should contain an AggregationOp";
    PGX_INFO("Agg node translated successfully with proper structure");
}

TEST_F(PlanNodeTranslationTest, TranslatesSortNode) {
    PGX_INFO("Testing Sort node translation");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Sort node with SeqScan as child
    Sort sort{};
    sort.plan.type = T_Sort;
    sort.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    sort.plan.righttree = nullptr;
    sort.numCols = 1;
    
    // Setup sort columns
    AttrNumber sortCols[] = {1};
    sort.sortColIdx = sortCols;
    Oid sortOps[] = {97}; // < operator for ascending
    sort.sortOperators = sortOps;
    bool nullsFirst[] = {false};
    sort.nullsFirst = nullsFirst;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&sort);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Sort translation should produce a module";
    
    // Verify the module contains expected operations
    bool hasSortOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.sort") {
            hasSortOp = true;
            PGX_INFO("Found SortOp in generated MLIR");
        }
    });
    
    EXPECT_TRUE(hasSortOp) << "Module should contain a SortOp";
    PGX_INFO("Sort node translated successfully with proper structure");
}

TEST_F(PlanNodeTranslationTest, TranslatesLimitNode) {
    PGX_INFO("Testing Limit node translation");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create a Const node for limit count
    Const limitConst{};
    limitConst.node.type = T_Const;
    limitConst.consttype = 23; // INT4OID
    limitConst.constvalue = 20; // Limit 20 rows
    limitConst.constisnull = false;
    limitConst.constbyval = true;
    
    // Create Limit node with SeqScan as child
    Limit limit{};
    limit.plan.type = T_Limit;
    limit.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    limit.plan.righttree = nullptr;
    limit.limitCount = reinterpret_cast<Node*>(&limitConst);
    limit.limitOffset = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&limit);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Limit translation should produce a module";
    
    // Verify the module contains expected operations
    bool hasLimitOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.limit") {
            hasLimitOp = true;
            PGX_INFO("Found LimitOp in generated MLIR");
            
            // Verify limit count attribute
            if (auto limitAttr = op->getAttrOfType<mlir::IntegerAttr>("count")) {
                EXPECT_EQ(limitAttr.getInt(), 20) << "Limit count should be 20";
            }
        }
    });
    
    EXPECT_TRUE(hasLimitOp) << "Module should contain a LimitOp";
    PGX_INFO("Limit node translated successfully with actual limit value");
}

TEST_F(PlanNodeTranslationTest, HandlesInvalidPlanNode) {
    PGX_INFO("Testing invalid plan node handling");
    
    // Create a plan node with invalid type
    Plan invalidPlan{};
    invalidPlan.type = -1; // Invalid type
    invalidPlan.lefttree = nullptr;
    invalidPlan.righttree = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = &invalidPlan;
    stmt.rtable = nullptr;
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should return nullptr for invalid plan
    ASSERT_EQ(module, nullptr) << "Invalid plan should not produce a module";
    PGX_INFO("Invalid plan node handled correctly");
}

TEST_F(PlanNodeTranslationTest, HandlesNullPlanTree) {
    PGX_INFO("Testing null plan tree handling");
    
    // Create PlannedStmt with null plan tree
    PlannedStmt stmt{};
    stmt.planTree = nullptr;
    stmt.rtable = nullptr;
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should return nullptr for null plan tree
    ASSERT_EQ(module, nullptr) << "Null plan tree should not produce a module";
    PGX_INFO("Null plan tree handled correctly");
}

TEST_F(PlanNodeTranslationTest, HandlesUnsupportedPlanType) {
    PGX_INFO("Testing unsupported plan type handling");
    
    // Create a plan node with unsupported but valid type
    Plan unsupportedPlan{};
    unsupportedPlan.type = 999; // Valid range but unsupported
    unsupportedPlan.lefttree = nullptr;
    unsupportedPlan.righttree = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = &unsupportedPlan;
    stmt.rtable = nullptr;
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should return nullptr for unsupported plan type
    ASSERT_EQ(module, nullptr) << "Unsupported plan type should not produce a module";
    PGX_INFO("Unsupported plan type handled correctly");
}

TEST_F(PlanNodeTranslationTest, TranslatesGatherNode) {
    PGX_INFO("Testing Gather node translation");
    
    // Create child Agg node
    Agg agg{};
    agg.plan.type = T_Agg;
    agg.plan.lefttree = nullptr;
    agg.plan.righttree = nullptr;
    agg.aggstrategy = AGG_HASHED;
    agg.numCols = 0; // No group by for this test
    
    // Create SeqScan as child of Agg
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    agg.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    
    // Create Gather node with Agg as child
    Gather gather{};
    gather.plan.type = T_Gather;
    gather.plan.lefttree = reinterpret_cast<Plan*>(&agg);
    gather.plan.righttree = nullptr;
    gather.num_workers = 2;
    gather.single_copy = false;
    gather.invisible = false;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&gather);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Gather translation should produce a module";
    
    // Since Gather is pass-through, verify we get the Agg operation
    bool hasAggOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.aggregation") {
            hasAggOp = true;
            PGX_INFO("Found AggregationOp from Gather's child");
        }
    });
    
    EXPECT_TRUE(hasAggOp) << "Module should contain the child AggregationOp";
    PGX_INFO("Gather node translated successfully (pass-through with workers=" + 
             std::to_string(gather.num_workers) + ")");
}

TEST_F(PlanNodeTranslationTest, TranslatesAggWithoutGroupBy) {
    PGX_INFO("Testing Agg node without GROUP BY columns");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Agg node with no GROUP BY
    Agg agg{};
    agg.plan.type = T_Agg;
    agg.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    agg.plan.righttree = nullptr;
    agg.aggstrategy = AGG_PLAIN;
    agg.numCols = 0; // No GROUP BY columns
    agg.grpColIdx = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&agg);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Agg without GROUP BY should produce a module";
    
    // Verify the AggregationOp was created
    bool hasAggOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.aggregation") {
            hasAggOp = true;
            // Verify it has empty group by columns
            if (auto groupByAttr = op->getAttrOfType<mlir::ArrayAttr>("group_by_cols")) {
                EXPECT_EQ(groupByAttr.size(), 0u) << "Should have no GROUP BY columns";
            }
            PGX_INFO("Found AggregationOp without GROUP BY columns");
        }
    });
    
    EXPECT_TRUE(hasAggOp) << "Module should contain an AggregationOp";
    PGX_INFO("Agg node without GROUP BY translated successfully");
}

TEST_F(PlanNodeTranslationTest, TranslatesSortWithMultipleColumns) {
    PGX_INFO("Testing Sort node with multiple columns");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Sort node with multiple sort columns
    Sort sort{};
    sort.plan.type = T_Sort;
    sort.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    sort.plan.righttree = nullptr;
    sort.numCols = 3;
    
    // Setup multiple sort columns
    AttrNumber sortCols[] = {1, 3, 2};
    sort.sortColIdx = sortCols;
    Oid sortOps[] = {97, 521, 97}; // <, >, < (mix of ascending/descending)
    sort.sortOperators = sortOps;
    bool nullsFirst[] = {false, true, false};
    sort.nullsFirst = nullsFirst;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&sort);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Sort with multiple columns should produce a module";
    
    // Verify the SortOp was created with multiple columns
    bool hasSortOp = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "relalg.sort") {
            hasSortOp = true;
            // Verify it has multiple sort specifications
            if (auto sortSpecsAttr = op->getAttrOfType<mlir::ArrayAttr>("sort_specs")) {
                EXPECT_EQ(sortSpecsAttr.size(), 3u) << "Should have 3 sort columns";
            }
            PGX_INFO("Found SortOp with multiple columns");
        }
    });
    
    EXPECT_TRUE(hasSortOp) << "Module should contain a SortOp";
    PGX_INFO("Sort node with multiple columns translated successfully");
}

TEST_F(PlanNodeTranslationTest, TranslatesComplexPlanTree) {
    PGX_INFO("Testing complex plan tree translation");
    
    // Create bottom-level SeqScan
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Sort with SeqScan as child
    Sort sort{};
    sort.plan.type = T_Sort;
    sort.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    sort.plan.righttree = nullptr;
    sort.numCols = 1;
    AttrNumber sortCols[] = {2};
    sort.sortColIdx = sortCols;
    
    // Create a Const node for limit
    Const limitConst{};
    limitConst.node.type = T_Const;
    limitConst.consttype = 23; // INT4OID
    limitConst.constvalue = 5;
    limitConst.constisnull = false;
    
    // Create Limit with Sort as child
    Limit limit{};
    limit.plan.type = T_Limit;
    limit.plan.lefttree = reinterpret_cast<Plan*>(&sort);
    limit.plan.righttree = nullptr;
    limit.limitCount = reinterpret_cast<Node*>(&limitConst);
    limit.limitOffset = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.planTree = reinterpret_cast<Plan*>(&limit);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Complex plan tree translation should produce a module";
    
    // Verify the module contains all expected operations
    bool hasSeqScan = false;
    bool hasSort = false;
    bool hasLimit = false;
    
    module->walk([&](mlir::Operation* op) {
        auto opName = op->getName().getStringRef();
        if (opName == "relalg.basetable") {
            hasSeqScan = true;
        } else if (opName == "relalg.sort") {
            hasSort = true;
        } else if (opName == "relalg.limit") {
            hasLimit = true;
        }
    });
    
    EXPECT_TRUE(hasSeqScan) << "Module should contain a BaseTableOp";
    EXPECT_TRUE(hasSort) << "Module should contain a SortOp";
    EXPECT_TRUE(hasLimit) << "Module should contain a LimitOp";
    
    PGX_INFO("Complex plan tree (Limit->Sort->SeqScan) translated successfully with all operations");
}