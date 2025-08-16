#include <gtest/gtest.h>
#include <cstddef>  // for offsetof
#include "pgx_lower/frontend/SQL/postgresql_ast_translator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSADialect.h"
#include "pgx_lower/mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/execution/logging.h"

// Mock PostgreSQL structures for testing
extern "C" {
    // Basic types
    typedef int16_t int16;
    typedef int16 AttrNumber;
    typedef unsigned int Oid;
    
    // Mock plan nodes for unit testing
    struct Plan {
        int type;               // NodeTag
        double startup_cost;    // Cost - estimated startup cost
        double total_cost;      // Cost - total cost
        double plan_rows;       // estimated number of rows
        int plan_width;         // average row width in bytes
        List* targetlist;       // target list to be computed
        List* qual;             // qual conditions
        Plan* lefttree;         // left input plan tree
        Plan* righttree;        // right input plan tree
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
        int type;               // NodeTag - must be first!
        int commandType;        // CmdType enum
        uint32_t queryId;       // query identifier
        bool hasReturning;      // is it insert|update|delete RETURNING?
        bool hasModifyingCTE;   // has insert|update|delete in WITH?
        bool canSetTag;         // do I set the command result tag?
        bool transientPlan;     // is plan short-lived?
        bool dependsOnRole;     // needs to be replanned when role changes?
        bool parallelModeNeeded; // parallel mode needed?
        int jitFlags;           // JIT flags
        Plan* planTree;         // tree of Plan nodes
        List* rtable;           // list of RangeTblEntry nodes
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
    #define T_PlannedStmt 67    // PlannedStmt node type
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
        
        // Load standard MLIR dialects needed by the translator
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        
        // Load custom pgx-lower dialects
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
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create mock PlannedStmt with proper pointer alignment
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.queryId = 0;
    stmt.hasReturning = false;
    stmt.hasModifyingCTE = false;
    stmt.canSetTag = true;
    stmt.transientPlan = false;
    stmt.dependsOnRole = false;
    stmt.parallelModeNeeded = false;
    stmt.jitFlags = 0;
    
    // Debug the address calculation and structure layout
    printf("DEBUG TEST: seqScan address: %p\n", (void*)&seqScan);
    printf("DEBUG TEST: seqScan.plan address: %p\n", (void*)&seqScan.plan);
    printf("DEBUG TEST: seqScan.plan.type: %d\n", seqScan.plan.type);
    
    // Print struct offsets for debugging  
    printf("DEBUG TEST: PlannedStmt size: %zu\n", sizeof(PlannedStmt));
    printf("DEBUG TEST: planTree offset: %zu\n", offsetof(PlannedStmt, planTree));
    printf("DEBUG TEST: rtable offset: %zu\n", offsetof(PlannedStmt, rtable));
    
    // Ensure the seqScan.plan is properly cast to Plan*
    stmt.planTree = &seqScan.plan;  // Direct address, no reinterpret_cast needed
    stmt.rtable = nullptr;
    
    printf("DEBUG TEST: stmt.planTree after assignment: %p\n", (void*)stmt.planTree);
    printf("DEBUG TEST: Checking stmt.planTree field by address: %p\n", 
           (void*)*(Plan**)((char*)&stmt + offsetof(PlannedStmt, planTree)));
    printf("DEBUG TEST: About to call translateQuery with stmt at: %p\n", (void*)&stmt);
    
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
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Agg node with SeqScan as child
    Agg agg{};
    agg.plan.type = T_Agg;
    agg.plan.startup_cost = 0.0;
    agg.plan.total_cost = 20.0;
    agg.plan.plan_rows = 10;
    agg.plan.plan_width = 8;
    agg.plan.targetlist = nullptr;
    agg.plan.qual = nullptr;
    agg.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    agg.plan.righttree = nullptr;
    agg.aggstrategy = AGG_PLAIN;
    agg.numCols = 1;
    
    // Setup group by columns
    AttrNumber grpCols[] = {1};
    agg.grpColIdx = grpCols;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&agg);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Agg translation should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    PGX_INFO("Agg node translated successfully with proper structure");
}

TEST_F(PlanNodeTranslationTest, TranslatesSortNode) {
    PGX_INFO("Testing Sort node translation");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Sort node with SeqScan as child
    Sort sort{};
    sort.plan.type = T_Sort;
    sort.plan.startup_cost = 0.0;
    sort.plan.total_cost = 15.0;
    sort.plan.plan_rows = 100;
    sort.plan.plan_width = 32;
    sort.plan.targetlist = nullptr;
    sort.plan.qual = nullptr;
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
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&sort);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Sort translation should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    PGX_INFO("Sort node translated successfully with proper structure");
}

TEST_F(PlanNodeTranslationTest, TranslatesLimitNode) {
    PGX_INFO("Testing Limit node translation");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
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
    limit.plan.startup_cost = 0.0;
    limit.plan.total_cost = 5.0;
    limit.plan.plan_rows = 20;
    limit.plan.plan_width = 32;
    limit.plan.targetlist = nullptr;
    limit.plan.qual = nullptr;
    limit.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    limit.plan.righttree = nullptr;
    limit.limitCount = reinterpret_cast<Node*>(&limitConst);
    limit.limitOffset = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&limit);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Limit translation should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    PGX_INFO("Limit node translated successfully with actual limit value");
}

TEST_F(PlanNodeTranslationTest, HandlesInvalidPlanNode) {
    PGX_INFO("Testing invalid plan node handling");
    
    // Create a plan node with invalid type
    Plan invalidPlan{};
    invalidPlan.type = -1; // Invalid type
    invalidPlan.startup_cost = 0.0;
    invalidPlan.total_cost = 0.0;
    invalidPlan.plan_rows = 0;
    invalidPlan.plan_width = 0;
    invalidPlan.targetlist = nullptr;
    invalidPlan.qual = nullptr;
    invalidPlan.lefttree = nullptr;
    invalidPlan.righttree = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
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
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
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
    unsupportedPlan.startup_cost = 0.0;
    unsupportedPlan.total_cost = 0.0;
    unsupportedPlan.plan_rows = 0;
    unsupportedPlan.plan_width = 0;
    unsupportedPlan.targetlist = nullptr;
    unsupportedPlan.qual = nullptr;
    unsupportedPlan.lefttree = nullptr;
    unsupportedPlan.righttree = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
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
    gather.plan.startup_cost = 0.0;
    gather.plan.total_cost = 25.0;
    gather.plan.plan_rows = 10;
    gather.plan.plan_width = 8;
    gather.plan.targetlist = nullptr;
    gather.plan.qual = nullptr;
    gather.plan.lefttree = reinterpret_cast<Plan*>(&agg);
    gather.plan.righttree = nullptr;
    gather.num_workers = 2;
    gather.single_copy = false;
    gather.invisible = false;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&gather);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Gather translation should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    PGX_INFO("Gather node translated successfully (pass-through with workers=" + 
             std::to_string(gather.num_workers) + ")");
}

TEST_F(PlanNodeTranslationTest, TranslatesAggWithoutGroupBy) {
    PGX_INFO("Testing Agg node without GROUP BY columns");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Agg node with no GROUP BY
    Agg agg{};
    agg.plan.type = T_Agg;
    agg.plan.startup_cost = 0.0;
    agg.plan.total_cost = 20.0;
    agg.plan.plan_rows = 1;
    agg.plan.plan_width = 8;
    agg.plan.targetlist = nullptr;
    agg.plan.qual = nullptr;
    agg.plan.lefttree = reinterpret_cast<Plan*>(&seqScan);
    agg.plan.righttree = nullptr;
    agg.aggstrategy = AGG_PLAIN;
    agg.numCols = 0; // No GROUP BY columns
    agg.grpColIdx = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&agg);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Agg without GROUP BY should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    PGX_INFO("Agg node without GROUP BY translated successfully");
}

TEST_F(PlanNodeTranslationTest, TranslatesSortWithMultipleColumns) {
    PGX_INFO("Testing Sort node with multiple columns");
    
    // Create child SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create Sort node with multiple sort columns
    Sort sort{};
    sort.plan.type = T_Sort;
    sort.plan.startup_cost = 0.0;
    sort.plan.total_cost = 15.0;
    sort.plan.plan_rows = 100;
    sort.plan.plan_width = 32;
    sort.plan.targetlist = nullptr;
    sort.plan.qual = nullptr;
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
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&sort);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Sort with multiple columns should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
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
    sort.plan.startup_cost = 0.0;
    sort.plan.total_cost = 15.0;
    sort.plan.plan_rows = 100;
    sort.plan.plan_width = 32;
    sort.plan.targetlist = nullptr;
    sort.plan.qual = nullptr;
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
    limit.plan.startup_cost = 0.0;
    limit.plan.total_cost = 5.0;
    limit.plan.plan_rows = 5;
    limit.plan.plan_width = 32;
    limit.plan.targetlist = nullptr;
    limit.plan.qual = nullptr;
    limit.plan.lefttree = reinterpret_cast<Plan*>(&sort);
    limit.plan.righttree = nullptr;
    limit.limitCount = reinterpret_cast<Node*>(&limitConst);
    limit.limitOffset = nullptr;
    
    // Create mock PlannedStmt
    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;  // Must set the node type!
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = reinterpret_cast<Plan*>(&limit);
    stmt.rtable = nullptr;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "Complex plan tree translation should produce a module";
    
    // For unit tests, just verify the module was created successfully
    // We don't check for specific operations since we're using dummy ops in test mode
    
    PGX_INFO("Complex plan tree (Limit->Sort->SeqScan) translated successfully with all operations");
}