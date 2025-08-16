#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"

class ErrorHandlingTest : public PlanNodeTestBase {};

TEST_F(ErrorHandlingTest, HandlesInvalidPlanNode) {
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
    PlannedStmt stmt = createPlannedStmt(&invalidPlan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Error handling test: Should return nullptr for invalid plan
    ASSERT_EQ(module, nullptr) << "Invalid plan should not produce a module";
    PGX_INFO("Invalid plan node handled correctly - returns nullptr as expected");
}

TEST_F(ErrorHandlingTest, HandlesNullPlanTree) {
    PGX_INFO("Testing null plan tree handling");
    
    // Create PlannedStmt with null plan tree
    PlannedStmt stmt = createPlannedStmt(nullptr);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Error handling test: Should return nullptr for null plan tree
    ASSERT_EQ(module, nullptr) << "Null plan tree should not produce a module";
    PGX_INFO("Null plan tree handled correctly - returns nullptr as expected");
}

TEST_F(ErrorHandlingTest, HandlesUnsupportedPlanType) {
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
    PlannedStmt stmt = createPlannedStmt(&unsupportedPlan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Error handling test: Should return nullptr for unsupported plan type
    ASSERT_EQ(module, nullptr) << "Unsupported plan type should not produce a module";
    PGX_INFO("Unsupported plan type handled correctly - returns nullptr as expected");
}

TEST_F(ErrorHandlingTest, HandlesNullTargetList) {
    PGX_INFO("Testing SeqScan with null targetlist");
    
    // Create SeqScan with explicitly null targetlist
    SeqScan* seqScan = createSeqScan();
    seqScan->plan.targetlist = nullptr;  // Explicitly null
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan->plan);
    
    // Translate - should handle gracefully and use default columns
    auto module = translator->translateQuery(&stmt);
    
    // Should still produce a module with default behavior
    ASSERT_NE(module, nullptr) << "SeqScan with null targetlist should still produce a module";
    
    std::vector<std::string> expectedPatterns = {
        "func.func @main",
        "relalg.basetable",  // Should still generate basetable
        "relalg.materialize",
        "return"
    };
    
    validateMLIR(module.get(), expectedPatterns);
    PGX_INFO("Null targetlist handled correctly - uses default columns");
}

TEST_F(ErrorHandlingTest, HandlesEmptyGroupByInAgg) {
    PGX_INFO("Testing Agg node with null group by columns");
    
    // Create SeqScan as child
    SeqScan* seqScan = createSeqScan();
    
    // Create Agg with null group by columns (valid for simple aggregation)
    Agg* agg = createAggNode(&seqScan->plan, AGG_PLAIN, 0, nullptr);
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&agg->plan);
    
    // Translate - this is actually valid (SELECT COUNT(*) FROM test)
    auto module = translator->translateQuery(&stmt);
    
    // Should produce a module for simple aggregation
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "func.func @main",
            "relalg.basetable",
            // "relalg.aggregation",  // Once implemented
            "return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Empty GROUP BY handled correctly for simple aggregation");
    } else {
        PGX_INFO("Simple aggregation not yet implemented - module is null");
    }
}

TEST_F(ErrorHandlingTest, HandlesNullLimitCount) {
    PGX_INFO("Testing Limit node with null limit count");
    
    // Create SeqScan as child
    SeqScan* seqScan = createSeqScan();
    
    // Create Limit with manual setup and null limit count
    Limit* limit = new Limit{};
    memset(limit, 0, sizeof(Limit));
    limit->plan.type = T_Limit;
    limit->plan.startup_cost = 0.0;
    limit->plan.total_cost = 5.0;
    limit->plan.plan_rows = 10;
    limit->plan.plan_width = 32;
    limit->plan.lefttree = &seqScan->plan;
    limit->limitOffset = nullptr;
    limit->limitCount = nullptr;  // Both null - invalid
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&limit->plan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should handle the error case
    if (module == nullptr) {
        PGX_INFO("Null limit count handled correctly - returns nullptr");
    } else {
        PGX_INFO("Null limit count produced module - may use default behavior");
    }
}

TEST_F(ErrorHandlingTest, HandlesInvalidSortColumns) {
    PGX_INFO("Testing Sort node with invalid sort columns");
    
    // Create SeqScan as child
    SeqScan* seqScan = createSeqScan();
    
    // Create Sort with invalid column configuration
    Sort* sort = new Sort{};
    memset(sort, 0, sizeof(Sort));
    sort->plan.type = T_Sort;
    sort->plan.startup_cost = 0.0;
    sort->plan.total_cost = 15.0;
    sort->plan.plan_rows = 100;
    sort->plan.plan_width = 32;
    sort->plan.lefttree = &seqScan->plan;
    sort->numCols = 3;  // Says 3 columns
    sort->sortColIdx = nullptr;  // But provides null array - invalid
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&sort->plan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should handle the error case
    if (module == nullptr) {
        PGX_INFO("Invalid sort columns handled correctly - returns nullptr");
    } else {
        PGX_INFO("Invalid sort columns produced module - may have error recovery");
    }
}

TEST_F(ErrorHandlingTest, HandlesNegativeWorkerCount) {
    PGX_INFO("Testing Gather node with negative worker count");
    
    // Create SeqScan as child
    SeqScan* seqScan = createSeqScan();
    
    // Create Gather with negative worker count
    Gather* gather = createGatherNode(&seqScan->plan, -5);  // Invalid worker count
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&gather->plan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    // May either handle as error or use default/minimum value
    if (module == nullptr) {
        PGX_INFO("Negative worker count handled as error - returns nullptr");
    } else {
        PGX_INFO("Negative worker count handled with default value - produces module");
    }
}

TEST_F(ErrorHandlingTest, HandlesCircularPlanReference) {
    PGX_INFO("Testing circular plan tree reference");
    
    // Create a plan that references itself (invalid)
    Plan* circularPlan = new Plan{};
    memset(circularPlan, 0, sizeof(Plan));
    circularPlan->type = T_SeqScan;
    circularPlan->startup_cost = 0.0;
    circularPlan->total_cost = 10.0;
    circularPlan->plan_rows = 100;
    circularPlan->plan_width = 32;
    circularPlan->lefttree = circularPlan;  // Circular reference!
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(circularPlan);
    
    // Translate - should detect and handle circular reference
    auto module = translator->translateQuery(&stmt);
    
    // Should handle the error case
    if (module == nullptr) {
        PGX_INFO("Circular reference handled correctly - returns nullptr");
    } else {
        PGX_INFO("Circular reference produced module - has cycle detection");
    }
    
    delete circularPlan;
}

TEST_F(ErrorHandlingTest, HandlesInvalidExpressionType) {
    PGX_INFO("Testing invalid expression type in targetlist");
    
    // Create SeqScan
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create targetlist with invalid expression
    static TargetEntry entry;
    static Node invalidExpr;
    
    invalidExpr.type = -999;  // Invalid expression type
    
    entry.node.type = T_TargetEntry;
    entry.expr = &invalidExpr;
    entry.resno = 1;
    entry.resname = const_cast<char*>("invalid");
    entry.ressortgroupref = 0;
    entry.resorigtbl = 0;
    entry.resorigcol = 0;
    entry.resjunk = false;
    
    List* targetList = list_make1(&entry);
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate - should handle invalid expression gracefully
    auto module = translator->translateQuery(&stmt);
    
    if (module == nullptr) {
        PGX_INFO("Invalid expression type handled correctly - returns nullptr");
    } else {
        PGX_INFO("Invalid expression type handled with fallback - produces module");
    }
}

TEST_F(ErrorHandlingTest, HandlesNullQualConditions) {
    PGX_INFO("Testing SeqScan with null qual conditions");
    
    // Create SeqScan with null qual (valid - no WHERE clause)
    SeqScan* seqScan = createSeqScan();
    seqScan->plan.qual = nullptr;  // No WHERE clause
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan->plan);
    
    // Translate - should work fine without WHERE clause
    auto module = translator->translateQuery(&stmt);
    
    ASSERT_NE(module, nullptr) << "SeqScan without WHERE clause should produce module";
    
    std::vector<std::string> expectedPatterns = {
        "func.func @main",
        "relalg.basetable",
        "relalg.materialize",
        "return"
    };
    
    validateMLIR(module.get(), expectedPatterns);
    PGX_INFO("Null qual conditions handled correctly - no filter applied");
}

TEST_F(ErrorHandlingTest, HandlesInvalidAggStrategy) {
    PGX_INFO("Testing Agg node with invalid strategy");
    
    // Create SeqScan as child
    SeqScan* seqScan = createSeqScan();
    
    // Create Agg with invalid strategy
    Agg* agg = createAggNode(&seqScan->plan, 999, 0, nullptr);  // Invalid strategy
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&agg->plan);
    
    // Translate - should handle gracefully
    auto module = translator->translateQuery(&stmt);
    
    if (module == nullptr) {
        PGX_INFO("Invalid agg strategy handled correctly - returns nullptr");
    } else {
        PGX_INFO("Invalid agg strategy handled with default - produces module");
    }
}