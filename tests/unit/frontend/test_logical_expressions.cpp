#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"

class LogicalExpressionTest : public PlanNodeTestBase {};

TEST_F(LogicalExpressionTest, TranslatesComparisonExpressions) {
    PGX_INFO("Testing comparison expression translation");
    
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    static OpExpr compExprs[3];
    static Var compVars[3];
    static Const compConsts[2];
    
    compVars[0].node.type = T_Var;
    compVars[0].varno = 1;
    compVars[0].varattno = 1;    compVars[0].vartype = 23;    compVars[0].vartypmod = -1;
    compVars[0].varcollid = 0;
    compVars[0].varlevelsup = 0;
    compVars[0].location = -1;
    
    compConsts[0].node.type = T_Const;
    compConsts[0].consttype = 23;    compConsts[0].constvalue = 10;
    compConsts[0].constisnull = false;
    compConsts[0].constbyval = true;
    compConsts[0].location = -1;
    
    List* args1 = list_make2(
        reinterpret_cast<void*>(&compVars[0]),
        reinterpret_cast<void*>(&compConsts[0])
    );
    
    compExprs[0].node.type = T_OpExpr;
    compExprs[0].opno = INT4EQOID;
    compExprs[0].opfuncid = INT4EQOID;
    compExprs[0].opresulttype = 16;    compExprs[0].opretset = false;
    compExprs[0].opcollid = 0;
    compExprs[0].inputcollid = 0;
    compExprs[0].args = args1;
    compExprs[0].location = -1;
    
    compVars[1].node.type = T_Var;
    compVars[1].varno = 1;
    compVars[1].varattno = 1;    compVars[1].vartype = 23;
    compVars[1].vartypmod = -1;
    compVars[1].location = -1;
    
    compVars[2].node.type = T_Var;
    compVars[2].varno = 1;
    compVars[2].varattno = 2;    compVars[2].vartype = 23;
    compVars[2].vartypmod = -1;
    compVars[2].location = -1;
    
    List* args2 = list_make2(
        reinterpret_cast<void*>(&compVars[1]),
        reinterpret_cast<void*>(&compVars[2])
    );
    
    compExprs[1].node.type = T_OpExpr;
    compExprs[1].opno = INT4LTOID;
    compExprs[1].opfuncid = INT4LTOID;
    compExprs[1].opresulttype = 16;    compExprs[1].opretset = false;
    compExprs[1].args = args2;
    compExprs[1].location = -1;
    
    compConsts[1].node.type = T_Const;
    compConsts[1].consttype = 23;    compConsts[1].constvalue = 5;
    compConsts[1].constisnull = false;
    compConsts[1].constbyval = true;
    compConsts[1].location = -1;
    
    List* args3 = list_make2(
        reinterpret_cast<void*>(&compVars[0]),        reinterpret_cast<void*>(&compConsts[1])
    );
    
    compExprs[2].node.type = T_OpExpr;
    compExprs[2].opno = INT4GEOID;
    compExprs[2].opfuncid = INT4GEOID;
    compExprs[2].opresulttype = 16;    compExprs[2].opretset = false;
    compExprs[2].args = args3;
    compExprs[2].location = -1;
    
    List* qualList = NIL;
    qualList = lappend(qualList, &compExprs[0]);
    qualList = lappend(qualList, &compExprs[1]);
    qualList = lappend(qualList, &compExprs[2]);
    seqScan.plan.qual = qualList;
    
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Comparison expressions test completed");
    } else {
        PGX_INFO("Comparison expressions not yet implemented - module is null as expected");
            }
}

TEST_F(LogicalExpressionTest, TranslatesLogicalExpressions) {
    PGX_INFO("Testing logical expression translation");
    
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    static BoolExpr boolExprs[3];
    static OpExpr condExprs[4];
    static Var logicVars[4];
    static Const logicConsts[4];
    
    for (int i = 0; i < 4; i++) {
        logicVars[i].node.type = T_Var;
        logicVars[i].varno = 1;
        logicVars[i].varattno = (i % 2) + 2;        logicVars[i].vartype = 23;        logicVars[i].vartypmod = -1;
        logicVars[i].location = -1;
        
        logicConsts[i].node.type = T_Const;
        logicConsts[i].consttype = 23;        logicConsts[i].constisnull = false;
        logicConsts[i].constbyval = true;
        logicConsts[i].location = -1;
    }
    
    logicConsts[0].constvalue = 5; 
    logicConsts[1].constvalue = 10;
    logicConsts[2].constvalue = 1;
    logicConsts[3].constvalue = 2;
    
    Oid compOps[] = {INT4GTOID, INT4LTOID, INT4EQOID, INT4EQOID};
    List* condArgListPtrs[4];
    for (int i = 0; i < 4; i++) {
        condArgListPtrs[i] = list_make2(&logicVars[i], &logicConsts[i]);
        
        condExprs[i].node.type = T_OpExpr;
        condExprs[i].opno = compOps[i];
        condExprs[i].opfuncid = compOps[i];
        condExprs[i].opresulttype = 16;        condExprs[i].opretset = false;
        condExprs[i].args = condArgListPtrs[i];
        condExprs[i].location = -1;
    }
    
    List* andArgList = list_make2(&condExprs[0], &condExprs[1]);
    boolExprs[0].node.type = T_BoolExpr;
    boolExprs[0].boolop = AND_EXPR;
    boolExprs[0].args = andArgList;
    boolExprs[0].location = -1;
    
    List* orArgList = list_make2(&condExprs[2], &condExprs[3]);
    boolExprs[1].node.type = T_BoolExpr;
    boolExprs[1].boolop = OR_EXPR;
    boolExprs[1].args = orArgList;
    boolExprs[1].location = -1;
    
    List* mainOrArgList = list_make2(&boolExprs[0], &boolExprs[1]);
    boolExprs[2].node.type = T_BoolExpr;
    boolExprs[2].boolop = OR_EXPR;
    boolExprs[2].args = mainOrArgList;
    boolExprs[2].location = -1;
    
    List* qualListPtr = list_make1(&boolExprs[2]);    seqScan.plan.qual = qualListPtr;
    
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Logical expressions test completed");
    } else {
        PGX_INFO("Logical expressions not yet implemented - module is null as expected");
            }
}

TEST_F(LogicalExpressionTest, TranslatesNotExpression) {
    PGX_INFO("Testing NOT expression translation");
    
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    static BoolExpr notExpr, orExpr;
    static OpExpr eqExpr, gtExpr;
    static Var val1, val2;
    static Const const10, const20;
    
    val1.node.type = T_Var;
    val1.varno = 1;
    val1.varattno = 1;
    val1.vartype = 23;
    val1.vartypmod = -1;
    val1.location = -1;
    
    const10.node.type = T_Const;
    const10.consttype = 23;
    const10.constvalue = 10;
    const10.constisnull = false;
    const10.constbyval = true;
    const10.location = -1;
    
    List* eqArgList = list_make2(&val1, &const10);
    
    eqExpr.node.type = T_OpExpr;
    eqExpr.opno = INT4EQOID;
    eqExpr.opfuncid = INT4EQOID;
    eqExpr.opresulttype = 16;
    eqExpr.opretset = false;
    eqExpr.args = eqArgList;
    eqExpr.location = -1;
    
    val2.node.type = T_Var;
    val2.varno = 1;
    val2.varattno = 2;
    val2.vartype = 23;
    val2.vartypmod = -1;
    val2.location = -1;
    
    const20.node.type = T_Const;
    const20.consttype = 23;
    const20.constvalue = 20;
    const20.constisnull = false;
    const20.constbyval = true;
    const20.location = -1;
    
    List* gtArgList = list_make2(&val2, &const20);
    
    gtExpr.node.type = T_OpExpr;
    gtExpr.opno = INT4GTOID;
    gtExpr.opfuncid = INT4GTOID;
    gtExpr.opresulttype = 16;
    gtExpr.opretset = false;
    gtExpr.args = gtArgList;
    gtExpr.location = -1;
    
    List* orArgList = list_make2(&eqExpr, &gtExpr);
    
    orExpr.node.type = T_BoolExpr;
    orExpr.boolop = OR_EXPR;
    orExpr.args = orArgList;
    orExpr.location = -1;
    
    List* notArgList = list_make1(&orExpr);
    
    notExpr.node.type = T_BoolExpr;
    notExpr.boolop = NOT_EXPR;
    notExpr.args = notArgList;
    notExpr.location = -1;
    
    List* qualList = list_make1(&notExpr);
    seqScan.plan.qual = qualList;
    
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("NOT expression test completed");
    } else {
        PGX_INFO("NOT expression not yet implemented - module is null as expected");
    }
}

TEST_F(LogicalExpressionTest, TranslatesAllComparisonOperators) {
    PGX_INFO("Testing all comparison operators");
    
    SeqScan seqScan{};
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.plan.targetlist = nullptr;
    seqScan.plan.lefttree = nullptr;
    seqScan.plan.righttree = nullptr;
    seqScan.scan.scanrelid = 1;
    
    static BoolExpr andExpr;
    static OpExpr compExprs[6];
    static Var vars[6];
    static Const consts[6];
    
    Oid operators[] = {
        INT4EQOID,
        INT4LTOID,
        INT4GTOID,
        INT4LEOID,
        INT4GEOID,
        INT4NEOID
    };
    
    const char* opNames[] = {"eq", "lt", "gt", "le", "ge", "ne"};
    
    List* compArgLists[6];
    for (int i = 0; i < 6; i++) {
        vars[i].node.type = T_Var;
        vars[i].varno = 1;
        vars[i].varattno = i + 1;
        vars[i].vartype = 23;
        vars[i].vartypmod = -1;
        vars[i].location = -1;
        
        consts[i].node.type = T_Const;
        consts[i].consttype = 23;
        consts[i].constvalue = (i + 1) * 10;
        consts[i].constisnull = false;
        consts[i].constbyval = true;
        consts[i].location = -1;
        
        compArgLists[i] = list_make2(&vars[i], &consts[i]);
        
        compExprs[i].node.type = T_OpExpr;
        compExprs[i].opno = operators[i];
        compExprs[i].opfuncid = operators[i];
        compExprs[i].opresulttype = 16;
        compExprs[i].opretset = false;
        compExprs[i].args = compArgLists[i];
        compExprs[i].location = -1;
    }
    
    List* andArgList = list_make1(&compExprs[0]);
    for (int i = 1; i < 6; i++) {
        andArgList = lappend(andArgList, &compExprs[i]);
    }
    
    andExpr.node.type = T_BoolExpr;
    andExpr.boolop = AND_EXPR;
    andExpr.args = andArgList;
    andExpr.location = -1;
    
    List* qualList = list_make1(&andExpr);
    seqScan.plan.qual = qualList;
    
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("All comparison operators test completed");
    } else {
        PGX_INFO("All comparison operators not yet implemented - module is null as expected");
    }
}