#include <gtest/gtest.h>
#include <cstddef>  // for offsetof
#include <cstring>  // for memset
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "pgx_lower/frontend/SQL/postgresql_ast_translator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSADialect.h"
#include "pgx_lower/mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/execution/logging.h"
#include "test_plan_node_helpers.h"

class MultiColumnBaseTableTest : public PlanNodeTestBase {};

TEST_F(MultiColumnBaseTableTest, TranslatesTwoColumnSeqScan) {
    PGX_INFO("Testing SeqScan with two columns (reproduces Test 4 bug)");
    
    // Create SeqScan node
    SeqScan* seqScan = createSeqScan();
    
    // Create targetlist with TWO columns: id (int4) and col2 (int4)
    // This simulates: SELECT id, col2 FROM test
    
    // Column 1: id
    Var* idVar = createVar(1, 1, 23);  // varno=1, varattno=1, type=INT4OID(23)
    TargetEntry* idEntry = createTargetEntry(
        reinterpret_cast<Node*>(idVar), 
        1,      // resno=1
        "id"    // resname
    );
    
    // Column 2: col2  
    Var* col2Var = createVar(1, 2, 23);  // varno=1, varattno=2, type=INT4OID(23)
    TargetEntry* col2Entry = createTargetEntry(
        reinterpret_cast<Node*>(col2Var),
        2,       // resno=2
        "col2"   // resname
    );
    
    // Create targetlist with both columns
    List* targetList = list_make2(idEntry, col2Entry);
    seqScan->plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan->plan);
    
    // Translate to RelAlg
    auto module = translator->translateQuery(&stmt);
    
    // Validate that the module was created
    ASSERT_NE(module, nullptr) << "Two-column SeqScan translation should produce a module";
    
    // Get the MLIR as string for detailed analysis
    std::string actualMLIR;
    llvm::raw_string_ostream stream(actualMLIR);
    module->print(stream);
    actualMLIR = stream.str();
    
    PGX_INFO("=== GENERATED MLIR FOR TWO-COLUMN SCAN ===");
    PGX_INFO(actualMLIR);
    PGX_INFO("=== END MLIR ===");
    
    // Critical validation: BOTH columns should be in the basetable operation
    std::vector<std::string> expectedPatterns = {
        "func.func @main",               // Main query function
        "relalg.basetable",              // BaseTableOp for table access
        "table_identifier = \"test",     // Table identifier
        "columns:",                      // Column definitions section
        "id =>",                         // First column: id
        "col2 =>",                       // Second column: col2 (THIS IS THE BUG!)
        "relalg.materialize",            // Materialize operation
        "[@test::@id,@test::@col2]",     // Both columns in materialize
        "return"                         // Function return
    };
    
    // Validate each expected pattern is present
    for (const auto& pattern : expectedPatterns) {
        if (actualMLIR.find(pattern) == std::string::npos) {
            PGX_ERROR("MISSING PATTERN: " + pattern);
            PGX_ERROR("This indicates the multi-column bug in PostgreSQL AST->RelAlg translation");
        }
        EXPECT_TRUE(actualMLIR.find(pattern) != std::string::npos) 
            << "Missing expected pattern: " << pattern 
            << "\nThis indicates the translator is not extracting all columns from targetlist";
    }
    
    // Specific validation for the critical bug
    if (actualMLIR.find("col2 =>") == std::string::npos) {
        PGX_ERROR("BUG REPRODUCED: col2 column missing from basetable operation!");
        PGX_ERROR("The translator is only extracting the first column from the targetlist.");
        PGX_ERROR("This explains why materialize operation fails - it references @test::@col2 that doesn't exist.");
    }
    
    PGX_INFO("Two-column SeqScan translation completed");
}

TEST_F(MultiColumnBaseTableTest, TranslatesThreeColumnSeqScan) {
    PGX_INFO("Testing SeqScan with three columns (extended multi-column test)");
    
    // Create SeqScan node
    SeqScan* seqScan = createSeqScan();
    
    // Create targetlist with THREE columns: id, col2, col3
    // This simulates: SELECT id, col2, col3 FROM test
    
    // Column 1: id
    Var* idVar = createVar(1, 1, 23);  // INT4
    TargetEntry* idEntry = createTargetEntry(
        reinterpret_cast<Node*>(idVar), 1, "id"
    );
    
    // Column 2: col2
    Var* col2Var = createVar(1, 2, 23);  // INT4
    TargetEntry* col2Entry = createTargetEntry(
        reinterpret_cast<Node*>(col2Var), 2, "col2"
    );
    
    // Column 3: col3 (different type to test type handling)
    Var* col3Var = createVar(1, 3, 25);  // TEXT type (OID 25)
    TargetEntry* col3Entry = createTargetEntry(
        reinterpret_cast<Node*>(col3Var), 3, "col3"
    );
    
    // Create targetlist with all three columns
    List* targetList = list_make1(idEntry);
    targetList = lappend(targetList, col2Entry);
    targetList = lappend(targetList, col3Entry);
    seqScan->plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan->plan);
    
    // Translate to RelAlg
    auto module = translator->translateQuery(&stmt);
    
    // Validate that the module was created
    ASSERT_NE(module, nullptr) << "Three-column SeqScan translation should produce a module";
    
    // Get the MLIR as string for analysis
    std::string actualMLIR;
    llvm::raw_string_ostream stream(actualMLIR);
    module->print(stream);
    actualMLIR = stream.str();
    
    PGX_INFO("=== GENERATED MLIR FOR THREE-COLUMN SCAN ===");
    PGX_INFO(actualMLIR);
    PGX_INFO("=== END MLIR ===");
    
    // Validate ALL three columns are present
    std::vector<std::string> expectedPatterns = {
        "relalg.basetable",              
        "id =>",                         // Column 1
        "col2 =>",                       // Column 2 
        "col3 =>",                       // Column 3
        "[@test::@id,@test::@col2,@test::@col3]"  // All three in materialize
    };
    
    for (const auto& pattern : expectedPatterns) {
        EXPECT_TRUE(actualMLIR.find(pattern) != std::string::npos) 
            << "Missing pattern for three-column test: " << pattern;
    }
    
    PGX_INFO("Three-column SeqScan translation completed");
}

TEST_F(MultiColumnBaseTableTest, ValidatesSingleColumnStillWorks) {
    PGX_INFO("Testing single column to ensure fix doesn't break existing functionality");
    
    // Create SeqScan node
    SeqScan* seqScan = createSeqScan();
    
    // Create targetlist with ONE column: id
    // This simulates: SELECT id FROM test (Tests 1-2 scenario)
    
    Var* idVar = createVar(1, 1, 23);
    TargetEntry* idEntry = createTargetEntry(
        reinterpret_cast<Node*>(idVar), 1, "id"
    );
    
    List* targetList = list_make1(idEntry);
    seqScan->plan.targetlist = targetList;
    
    // Create PlannedStmt
    PlannedStmt stmt = createPlannedStmt(&seqScan->plan);
    
    // Translate to RelAlg
    auto module = translator->translateQuery(&stmt);
    
    // Validate that the module was created
    ASSERT_NE(module, nullptr) << "Single-column SeqScan should still work";
    
    // Get the MLIR as string
    std::string actualMLIR;
    llvm::raw_string_ostream stream(actualMLIR);
    module->print(stream);
    actualMLIR = stream.str();
    
    // Validate single column works correctly
    std::vector<std::string> expectedPatterns = {
        "relalg.basetable",              
        "id =>",                         // Should have id column
        "[@test::@id]"                   // Only id in materialize
    };
    
    for (const auto& pattern : expectedPatterns) {
        EXPECT_TRUE(actualMLIR.find(pattern) != std::string::npos) 
            << "Single-column test failed for pattern: " << pattern;
    }
    
    // Should NOT contain col2 for single-column test
    EXPECT_TRUE(actualMLIR.find("col2 =>") == std::string::npos)
        << "Single-column test should not contain col2";
    
    PGX_INFO("Single-column validation passed");
}