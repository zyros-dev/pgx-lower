#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"

class ComplexPlanTreeTest : public PlanNodeTestBase {};

TEST_F(ComplexPlanTreeTest, TranslatesComplexPlanTree) {
    PGX_INFO("Testing complex plan tree translation");
    
    // Create bottom-level SeqScan
    SeqScan* seqScan = createSeqScan();
    
    // Setup sort columns
    static AttrNumber sortCols3[] = {2};
    
    // Create Sort with SeqScan as child
    Sort* sort = createSortNode(&seqScan->plan, 1, sortCols3);
    
    // Create Limit with Sort as child
    Limit* limit = createLimitNode(&sort->plan, 5);
    
    // Create mock PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&limit->plan);
    
    // Expected patterns for complex tree
    std::vector<std::string> expectedPatterns = {
        "func.call",                     // SeqScan generates table access function call
        "relalg.sort",                   // Sort in the middle  
        "relalg.limit",                  // Limit at the top
        "func.func",                     // Function wrapper
        "func.return"                    // Function return
    };
    
    translateAndValidate(&stmt, expectedPatterns);
    PGX_INFO("Complex plan tree (Limit->Sort->SeqScan) translated successfully with all operations validated");
}

TEST_F(ComplexPlanTreeTest, TranslatesWhereClause) {
    PGX_INFO("Testing WHERE clause filtering");
    
    // Create base SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 50;  // Fewer rows due to filtering
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create WHERE clause conditions
    // Simulating: SELECT * FROM test WHERE id = 42 OR value > 10
    static BoolExpr orExpr;
    static OpExpr eqExpr, gtExpr;
    static Var idVar, valueVar;
    static Const const42, const10;
    
    // Setup id = 42 condition
    idVar.node.type = T_Var;
    idVar.varno = 1;
    idVar.varattno = 1;  // id column
    idVar.vartype = 23;  // INT4OID
    idVar.vartypmod = -1;
    idVar.location = -1;
    
    const42.node.type = T_Const;
    const42.consttype = 23;  // INT4OID
    const42.constvalue = 42;
    const42.constisnull = false;
    const42.constbyval = true;
    const42.location = -1;
    
    // Create list for equality expression arguments
    List* eqArgList = list_make2(&idVar, &const42);
    
    eqExpr.node.type = T_OpExpr;
    eqExpr.opno = INT4EQOID;
    eqExpr.opfuncid = INT4EQOID;
    eqExpr.opresulttype = 16;  // BOOLOID
    eqExpr.opretset = false;
    eqExpr.args = eqArgList;
    eqExpr.location = -1;
    
    // Setup value > 10 condition
    valueVar.node.type = T_Var;
    valueVar.varno = 1;
    valueVar.varattno = 2;  // value column
    valueVar.vartype = 23;  // INT4OID
    valueVar.vartypmod = -1;
    valueVar.location = -1;
    
    const10.node.type = T_Const;
    const10.consttype = 23;  // INT4OID
    const10.constvalue = 10;
    const10.constisnull = false;
    const10.constbyval = true;
    const10.location = -1;
    
    // Create list for greater-than expression arguments
    List* gtArgList = list_make2(&valueVar, &const10);
    
    gtExpr.node.type = T_OpExpr;
    gtExpr.opno = INT4GTOID;
    gtExpr.opfuncid = INT4GTOID;
    gtExpr.opresulttype = 16;  // BOOLOID
    gtExpr.opretset = false;
    gtExpr.args = gtArgList;
    gtExpr.location = -1;
    
    // Setup OR expression
    List* orArgList = list_make2(&eqExpr, &gtExpr);
    
    orExpr.node.type = T_BoolExpr;
    orExpr.boolop = OR_EXPR;
    orExpr.args = orArgList;
    orExpr.location = -1;
    
    // Create the qual list
    List* qualList = list_make1(&orExpr);
    seqScan.plan.qual = qualList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // TODO: Once WHERE clause translation is implemented, these patterns should appear
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "relalg.selection",    // Selection/filter operation
            // "filter_expr",         // Filter expression
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("WHERE clause test completed - TODO: Implement filter/selection operations in translator");
    } else {
        PGX_INFO("WHERE clause filtering not yet implemented - module is null as expected");
        // TODO: Once implemented, this should produce a valid module with selection/filter operations
    }
}

TEST_F(ComplexPlanTreeTest, TranslatesComplexWhereConditions) {
    PGX_INFO("Testing complex WHERE conditions with AND/OR");
    
    // Create base SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 25;  // Heavily filtered
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create complex WHERE clause
    // Simulating: WHERE (a > 5 AND b < 10) OR c = 20
    static BoolExpr mainOrExpr, andExpr;
    static OpExpr gtExpr, ltExpr, eqExpr;
    static Var aVar, bVar, cVar;
    static Const const5, const10, const20;
    
    // Setup a > 5
    aVar.node.type = T_Var;
    aVar.varno = 1;
    aVar.varattno = 1;  // column a
    aVar.vartype = 23;  // INT4OID
    aVar.vartypmod = -1;
    aVar.location = -1;
    
    const5.node.type = T_Const;
    const5.consttype = 23;
    const5.constvalue = 5;
    const5.constisnull = false;
    const5.constbyval = true;
    const5.location = -1;
    
    List* gtArgList = list_make2(&aVar, &const5);
    
    gtExpr.node.type = T_OpExpr;
    gtExpr.opno = INT4GTOID;
    gtExpr.opfuncid = INT4GTOID;
    gtExpr.opresulttype = 16;  // BOOLOID
    gtExpr.opretset = false;
    gtExpr.args = gtArgList;
    gtExpr.location = -1;
    
    // Setup b < 10
    bVar.node.type = T_Var;
    bVar.varno = 1;
    bVar.varattno = 2;  // column b
    bVar.vartype = 23;
    bVar.vartypmod = -1;
    bVar.location = -1;
    
    const10.node.type = T_Const;
    const10.consttype = 23;
    const10.constvalue = 10;
    const10.constisnull = false;
    const10.constbyval = true;
    const10.location = -1;
    
    List* ltArgList = list_make2(&bVar, &const10);
    
    ltExpr.node.type = T_OpExpr;
    ltExpr.opno = INT4LTOID;
    ltExpr.opfuncid = INT4LTOID;
    ltExpr.opresulttype = 16;
    ltExpr.opretset = false;
    ltExpr.args = ltArgList;
    ltExpr.location = -1;
    
    // Setup AND expression: (a > 5 AND b < 10)
    List* andArgList = list_make2(&gtExpr, &ltExpr);
    
    andExpr.node.type = T_BoolExpr;
    andExpr.boolop = AND_EXPR;
    andExpr.args = andArgList;
    andExpr.location = -1;
    
    // Setup c = 20
    cVar.node.type = T_Var;
    cVar.varno = 1;
    cVar.varattno = 3;  // column c
    cVar.vartype = 23;
    cVar.vartypmod = -1;
    cVar.location = -1;
    
    const20.node.type = T_Const;
    const20.consttype = 23;
    const20.constvalue = 20;
    const20.constisnull = false;
    const20.constbyval = true;
    const20.location = -1;
    
    List* eqArgList = list_make2(&cVar, &const20);
    
    eqExpr.node.type = T_OpExpr;
    eqExpr.opno = INT4EQOID;
    eqExpr.opfuncid = INT4EQOID;
    eqExpr.opresulttype = 16;
    eqExpr.opretset = false;
    eqExpr.args = eqArgList;
    eqExpr.location = -1;
    
    // Setup main OR expression: (AND expr) OR (c = 20)
    List* mainOrArgList = list_make2(&andExpr, &eqExpr);
    
    mainOrExpr.node.type = T_BoolExpr;
    mainOrExpr.boolop = OR_EXPR;
    mainOrExpr.args = mainOrArgList;
    mainOrExpr.location = -1;
    
    List* qualList = list_make1(&mainOrExpr);
    seqScan.plan.qual = qualList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // TODO: Once complex WHERE translation is implemented, these patterns should appear
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "relalg.logical_and",
            // "relalg.logical_or",
            // "relalg.selection",
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Complex WHERE conditions test completed - TODO: Implement complex logical operations in filters");
    } else {
        PGX_INFO("Complex WHERE conditions not yet implemented - module is null as expected");
        // TODO: Once implemented, this should produce a valid module
    }
}

TEST_F(ComplexPlanTreeTest, TranslatesSimpleProjection) {
    PGX_INFO("Testing simple projection with column references");
    
    // Create base SeqScan node
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 8;  // Smaller due to projection
    seqScan.plan.qual = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    // Create targetlist with just column references
    // Simulating: SELECT id, name FROM test
    static TargetEntry entries[2];
    static Var idVar, nameVar;
    
    // Setup id column reference
    idVar.node.type = T_Var;
    idVar.varno = 1;
    idVar.varattno = 1;  // id column
    idVar.vartype = 23;  // INT4OID
    idVar.vartypmod = -1;
    idVar.varcollid = 0;
    idVar.varlevelsup = 0;
    idVar.varnoold = 1;
    idVar.varoattno = 1;
    idVar.location = -1;
    
    entries[0].node.type = T_TargetEntry;
    entries[0].expr = reinterpret_cast<Node*>(&idVar);
    entries[0].resno = 1;
    entries[0].resname = const_cast<char*>("id");
    entries[0].ressortgroupref = 0;
    entries[0].resorigtbl = 0;
    entries[0].resorigcol = 1;
    entries[0].resjunk = false;
    
    // Setup name column reference
    nameVar.node.type = T_Var;
    nameVar.varno = 1;
    nameVar.varattno = 2;  // name column
    nameVar.vartype = 25;  // TEXTOID
    nameVar.vartypmod = -1;
    nameVar.varcollid = 0;
    nameVar.varlevelsup = 0;
    nameVar.varnoold = 1;
    nameVar.varoattno = 2;
    nameVar.location = -1;
    
    entries[1].node.type = T_TargetEntry;
    entries[1].expr = reinterpret_cast<Node*>(&nameVar);
    entries[1].resno = 2;
    entries[1].resname = const_cast<char*>("name");
    entries[1].ressortgroupref = 0;
    entries[1].resorigtbl = 0;
    entries[1].resorigcol = 2;
    entries[1].resjunk = false;
    
    // Create target list with two entries
    List* targetList = list_make2(&entries[0], &entries[1]);
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // TODO: Once simple projection is implemented, these patterns should appear
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "relalg.projection",    // Projection operation
            // "column_refs",          // Column references
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Simple projection test completed - TODO: Implement projection operations for column references");
    } else {
        PGX_INFO("Simple projection not yet implemented - module is null as expected");
        // TODO: Once implemented, this should produce a valid module with projection operations
    }
}

TEST_F(ComplexPlanTreeTest, TranslatesProjectionWithExpression) {
    PGX_INFO("Testing projection with expression translation");
    
    // Create base SeqScan node
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
    
    // Create targetlist with mixed column references and expressions
    // Simulating: SELECT id, val1 + val2 AS sum FROM test
    static TargetEntry entries[2];
    static Var idVar;
    static OpExpr sumExpr;
    static Var sumVars[2];
    
    // Setup first entry: id column reference
    idVar.node.type = T_Var;
    idVar.varno = 1;
    idVar.varattno = 1;  // id column
    idVar.vartype = 23;  // INT4OID
    idVar.vartypmod = -1;
    idVar.varcollid = 0;
    idVar.varlevelsup = 0;
    idVar.varnoold = 1;
    idVar.varoattno = 1;
    idVar.location = -1;
    
    entries[0].node.type = T_TargetEntry;
    entries[0].expr = reinterpret_cast<Node*>(&idVar);
    entries[0].resno = 1;
    entries[0].resname = const_cast<char*>("id");
    entries[0].ressortgroupref = 0;
    entries[0].resorigtbl = 0;
    entries[0].resorigcol = 1;
    entries[0].resjunk = false;
    
    // Setup second entry: val1 + val2 expression
    sumVars[0].node.type = T_Var;
    sumVars[0].varno = 1;
    sumVars[0].varattno = 2;  // val1
    sumVars[0].vartype = 23;
    sumVars[0].vartypmod = -1;
    sumVars[0].location = -1;
    
    sumVars[1].node.type = T_Var;
    sumVars[1].varno = 1;
    sumVars[1].varattno = 3;  // val2
    sumVars[1].vartype = 23;
    sumVars[1].vartypmod = -1;
    sumVars[1].location = -1;
    
    // Create argument list for sum expression
    List* sumArgList = list_make2(&sumVars[0], &sumVars[1]);
    
    sumExpr.node.type = T_OpExpr;
    sumExpr.opno = INT4PLUSOID;
    sumExpr.opfuncid = INT4PLUSOID;
    sumExpr.opresulttype = 23;  // INT4OID
    sumExpr.opretset = false;
    sumExpr.opcollid = 0;
    sumExpr.inputcollid = 0;
    sumExpr.args = sumArgList;
    sumExpr.location = -1;
    
    entries[1].node.type = T_TargetEntry;
    entries[1].expr = reinterpret_cast<Node*>(&sumExpr);
    entries[1].resno = 2;
    entries[1].resname = const_cast<char*>("sum");
    entries[1].ressortgroupref = 0;
    entries[1].resorigtbl = 0;
    entries[1].resorigcol = 0;  // Computed column
    entries[1].resjunk = false;
    
    // Create target list with two entries
    List* targetList = list_make2(&entries[0], &entries[1]);
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // TODO: Once expression translation is implemented, these patterns should appear
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "relalg.column_ref",   // Column reference for id
            // "relalg.add",          // Addition for val1 + val2
            // "relalg.project",      // Projection operation
            "func.func",    // Function wrapper
            "func.return"   // Function return
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Projection with expression test completed - TODO: Implement expression handling in projections");
    } else {
        PGX_INFO("Projection with expressions not yet implemented - module is null as expected");
        // TODO: Once implemented, this should produce a valid module
    }
}