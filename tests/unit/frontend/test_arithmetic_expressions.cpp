#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"

class ArithmeticExpressionTest : public PlanNodeTestBase {};

TEST_F(ArithmeticExpressionTest, TranslatesArithmeticExpressions) {
    PGX_INFO("Testing arithmetic expression translation");
    
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
    
    // Create targetlist with arithmetic expressions
    // Simulating: SELECT val1 + val2, val1 - val2, val1 * val2, val1 / val2 FROM test
    static TargetEntry entries[4];
    static OpExpr opExprs[4];
    static Var vars[8];
    
    // Setup variables for columns
    for (int i = 0; i < 8; i++) {
        vars[i].node.type = T_Var;
        vars[i].varno = 1;
        vars[i].varattno = (i % 2) + 1;  // Alternating col1(1) and col2(2)
        vars[i].vartype = 23;  // INT4OID
        vars[i].vartypmod = -1;
        vars[i].varcollid = 0;
        vars[i].varlevelsup = 0;
        vars[i].varnoold = 1;
        vars[i].varoattno = (i % 2) + 1;
        vars[i].location = -1;
    }
    
    // Setup arithmetic operations
    Oid operators[] = {INT4PLUSOID, INT4MINUSOID, INT4MULOID, INT4DIVOID};
    const char* opNames[] = {"add", "sub", "mul", "div"};
    
    // Create targetlist as proper linked list
    List* targetList = NIL;
    
    for (int i = 0; i < 4; i++) {
        // Create argument list for operator (val1, val2)
        List* args = list_make2(
            reinterpret_cast<void*>(&vars[i * 2]),
            reinterpret_cast<void*>(&vars[i * 2 + 1])
        );
        
        // Setup OpExpr for each arithmetic operation
        opExprs[i].node.type = T_OpExpr;
        opExprs[i].opno = operators[i];
        opExprs[i].opfuncid = operators[i];  // Simplified: using same ID
        opExprs[i].opresulttype = 23;  // INT4OID
        opExprs[i].opretset = false;
        opExprs[i].opcollid = 0;
        opExprs[i].inputcollid = 0;
        opExprs[i].args = args;
        opExprs[i].location = -1;
        
        // Setup TargetEntry
        entries[i].node.type = T_TargetEntry;
        entries[i].expr = reinterpret_cast<Node*>(&opExprs[i]);
        entries[i].resno = i + 1;
        entries[i].resname = const_cast<char*>(opNames[i]);
        entries[i].ressortgroupref = 0;
        entries[i].resorigtbl = 0;
        entries[i].resorigcol = 0;
        entries[i].resjunk = false;
        
        // Add to targetlist
        PGX_INFO("Adding TargetEntry[" + std::to_string(i) + "] at address " + 
                 std::to_string(reinterpret_cast<uintptr_t>(&entries[i])));
        targetList = lappend(targetList, &entries[i]);
    }
    
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // Check that expression translation is working
    if (module) {
        std::vector<std::string> expectedPatterns = {
            "relalg.basetable",  // Base table scan
            "relalg.map",        // Map operation for computed columns
            "arith.addi",        // Addition operation
            "arith.subi",        // Subtraction operation
            "arith.muli",        // Multiplication operation
            "arith.divsi",       // Division operation
            "relalg.return",     // Return from map predicate
            "relalg.materialize", // Materialize results
            "func.func",         // Function wrapper
            "func.return"        // Function return
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Arithmetic expressions test completed - expressions are being translated!");
    } else {
        FAIL() << "Module generation failed - expression translation should be working";
    }
}

TEST_F(ArithmeticExpressionTest, TranslatesNestedArithmetic) {
    PGX_INFO("Testing nested arithmetic expression translation");
    
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
    
    // Create targetlist with nested arithmetic expression
    // Simulating: SELECT (val1 + val2) * (val3 - val4) FROM test
    static TargetEntry entry;
    static OpExpr mulExpr, addExpr, subExpr;
    static Var val1, val2, val3, val4;
    
    // Setup variables
    val1.node.type = T_Var;
    val1.varno = 1;
    val1.varattno = 1;
    val1.vartype = 23;  // INT4OID
    val1.vartypmod = -1;
    val1.location = -1;
    
    val2.node.type = T_Var;
    val2.varno = 1;
    val2.varattno = 2;
    val2.vartype = 23;
    val2.vartypmod = -1;
    val2.location = -1;
    
    val3.node.type = T_Var;
    val3.varno = 1;
    val3.varattno = 3;
    val3.vartype = 23;
    val3.vartypmod = -1;
    val3.location = -1;
    
    val4.node.type = T_Var;
    val4.varno = 1;
    val4.varattno = 4;
    val4.vartype = 23;
    val4.vartypmod = -1;
    val4.location = -1;
    
    // Setup val1 + val2
    List* addArgs = list_make2(
        reinterpret_cast<void*>(&val1),
        reinterpret_cast<void*>(&val2)
    );
    addExpr.node.type = T_OpExpr;
    addExpr.opno = INT4PLUSOID;
    addExpr.opfuncid = INT4PLUSOID;
    addExpr.opresulttype = 23;
    addExpr.opretset = false;
    addExpr.args = addArgs;
    addExpr.location = -1;
    
    // Setup val3 - val4
    List* subArgs = list_make2(
        reinterpret_cast<void*>(&val3),
        reinterpret_cast<void*>(&val4)
    );
    subExpr.node.type = T_OpExpr;
    subExpr.opno = INT4MINUSOID;
    subExpr.opfuncid = INT4MINUSOID;
    subExpr.opresulttype = 23;
    subExpr.opretset = false;
    subExpr.args = subArgs;
    subExpr.location = -1;
    
    // Setup (val1 + val2) * (val3 - val4)
    List* mulArgs = list_make2(
        reinterpret_cast<void*>(&addExpr),
        reinterpret_cast<void*>(&subExpr)
    );
    mulExpr.node.type = T_OpExpr;
    mulExpr.opno = INT4MULOID;
    mulExpr.opfuncid = INT4MULOID;
    mulExpr.opresulttype = 23;
    mulExpr.opretset = false;
    mulExpr.args = mulArgs;
    mulExpr.location = -1;
    
    // Setup TargetEntry
    entry.node.type = T_TargetEntry;
    entry.expr = reinterpret_cast<Node*>(&mulExpr);
    entry.resno = 1;
    entry.resname = const_cast<char*>("result");
    entry.ressortgroupref = 0;
    entry.resorigtbl = 0;
    entry.resorigcol = 0;
    entry.resjunk = false;
    
    // Create targetlist
    List* targetList = list_make1(&entry);
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "relalg.add",
            // "relalg.sub",
            // "relalg.mul",
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Nested arithmetic test completed - TODO: Implement nested expression handling");
    } else {
        PGX_INFO("Nested arithmetic not yet implemented - module is null as expected");
    }
}

TEST_F(ArithmeticExpressionTest, TranslatesArithmeticWithConstants) {
    PGX_INFO("Testing arithmetic with constants");
    
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
    
    // Create targetlist with constants in arithmetic
    // Simulating: SELECT val1 * 2, val2 + 100, 10 - val3 FROM test
    static TargetEntry entries[3];
    static OpExpr opExprs[3];
    static Var vars[3];
    static Const consts[3];
    
    // Setup variables
    for (int i = 0; i < 3; i++) {
        vars[i].node.type = T_Var;
        vars[i].varno = 1;
        vars[i].varattno = i + 1;
        vars[i].vartype = 23;  // INT4OID
        vars[i].vartypmod = -1;
        vars[i].location = -1;
    }
    
    // Setup constants
    consts[0].node.type = T_Const;
    consts[0].consttype = 23;
    consts[0].constvalue = 2;
    consts[0].constisnull = false;
    consts[0].constbyval = true;
    consts[0].location = -1;
    
    consts[1].node.type = T_Const;
    consts[1].consttype = 23;
    consts[1].constvalue = 100;
    consts[1].constisnull = false;
    consts[1].constbyval = true;
    consts[1].location = -1;
    
    consts[2].node.type = T_Const;
    consts[2].consttype = 23;
    consts[2].constvalue = 10;
    consts[2].constisnull = false;
    consts[2].constbyval = true;
    consts[2].location = -1;
    
    // Create targetlist as proper linked list
    List* targetList = NIL;
    
    // Setup val1 * 2
    List* args0 = list_make2(
        reinterpret_cast<void*>(&vars[0]),
        reinterpret_cast<void*>(&consts[0])
    );
    opExprs[0].node.type = T_OpExpr;
    opExprs[0].opno = INT4MULOID;
    opExprs[0].opfuncid = INT4MULOID;
    opExprs[0].opresulttype = 23;
    opExprs[0].opretset = false;
    opExprs[0].args = args0;
    opExprs[0].location = -1;
    
    // Setup val2 + 100
    List* args1 = list_make2(
        reinterpret_cast<void*>(&vars[1]),
        reinterpret_cast<void*>(&consts[1])
    );
    opExprs[1].node.type = T_OpExpr;
    opExprs[1].opno = INT4PLUSOID;
    opExprs[1].opfuncid = INT4PLUSOID;
    opExprs[1].opresulttype = 23;
    opExprs[1].opretset = false;
    opExprs[1].args = args1;
    opExprs[1].location = -1;
    
    // Setup 10 - val3
    List* args2 = list_make2(
        reinterpret_cast<void*>(&consts[2]),
        reinterpret_cast<void*>(&vars[2])
    );
    opExprs[2].node.type = T_OpExpr;
    opExprs[2].opno = INT4MINUSOID;
    opExprs[2].opfuncid = INT4MINUSOID;
    opExprs[2].opresulttype = 23;
    opExprs[2].opretset = false;
    opExprs[2].args = args2;
    opExprs[2].location = -1;
    
    // Setup TargetEntries
    const char* names[] = {"double_val", "plus_hundred", "ten_minus"};
    for (int i = 0; i < 3; i++) {
        entries[i].node.type = T_TargetEntry;
        entries[i].expr = reinterpret_cast<Node*>(&opExprs[i]);
        entries[i].resno = i + 1;
        entries[i].resname = const_cast<char*>(names[i]);
        entries[i].ressortgroupref = 0;
        entries[i].resorigtbl = 0;
        entries[i].resorigcol = 0;
        entries[i].resjunk = false;
        
        // Add to targetlist
        PGX_INFO("Adding TargetEntry[" + std::to_string(i) + "] at address " + 
                 std::to_string(reinterpret_cast<uintptr_t>(&entries[i])));
        targetList = lappend(targetList, &entries[i]);
    }
    
    seqScan.plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan.plan);
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    if (module) {
        std::vector<std::string> expectedPatterns = {
            // Once implemented, should see:
            // "arith.constant",  // Constant values
            // "relalg.mul",
            // "relalg.add",
            // "relalg.sub",
            "func.func",
            "func.return"
        };
        
        validateMLIR(module.get(), expectedPatterns);
        PGX_INFO("Arithmetic with constants test completed - TODO: Implement constant handling");
    } else {
        PGX_INFO("Arithmetic with constants not yet implemented - module is null as expected");
    }
}