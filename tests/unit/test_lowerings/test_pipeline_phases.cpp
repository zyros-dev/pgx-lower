#include <gtest/gtest.h>
#include "standalone_mlir_runner.h"
#include "pgx-lower/utility/logging.h"
#include <cstdlib>

using namespace pgx_test;

class MLIRLoweringPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pgx_lower::log::log_enable = true;
        pgx_lower::log::log_debug = true;
        pgx_lower::log::log_io = true;
        pgx_lower::log::log_trace = true;
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::GENERAL);
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::RELALG_LOWER);
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::DB_LOWER);
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::DSA_LOWER);
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::UTIL_LOWER);
        pgx_lower::log::enabled_categories.insert(pgx_lower::log::Category::JIT);
        
        tester = std::make_unique<StandalonePipelineTester>();
    }
    
    void TearDown() override {
        tester.reset();
    }
    
    std::unique_ptr<StandalonePipelineTester> tester;
};

TEST_F(MLIRLoweringPipelineTest, SimpleConstantPipeline) {
    // Very basic test - just constant return
    auto simpleMLIR = R"(
        module {
            func.func @main() -> i32 {
                %c42 = arith.constant 42 : i32
                return %c42 : i32
            }
        }
    )";
    
    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) 
        << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a failed";
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b failed";
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3b failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";
    
    // Check that we got LLVM operations in the final result
    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) 
        << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, TestOne) {
    // Very basic test - just constant return
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {column_order = ["id"], table_identifier = "test|oid:11295606"} columns: {id => @test::@id({type = i32})}
            %1 = relalg.materialize %0 [@test::@id] => ["id"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR))
        << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a() && tester->runPhase3b() && tester->runPhase3c())
        << "Complete pipeline failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    // Check that we got LLVM operations in the final result
    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos)
        << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, Test9Arith) {
    // Very basic test - just constant return
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {table_identifier = "test_arithmetic|oid:8338330"} columns: {id => @test_arithmetic::@id({type = i32}), val1 => @test_arithmetic::@val1({type = i32}), val2 => @test_arithmetic::@val2({type = i32})}
            %1 = relalg.map %0 computes : [@map::@addition({type = i32})] (%arg0: !relalg.tuple){
              %3 = relalg.getcol %arg0 @test_arithmetic::@val1 : i32
              %4 = relalg.getcol %arg0 @test_arithmetic::@val2 : i32
              %5 = db.add %3 : i32, %4 : i32
              relalg.return %5 : i32
            }
            %2 = relalg.materialize %1 [@map::@addition] => ["addition"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a failed";
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b failed";
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, Test11) {
    // This replicates the relalg from test 11. If you want to change this input, you need to change the AST parser
    // to produce that as well.
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {table_identifier = "test_logical|oid:8493992"} columns: {flag1 => @test_logical::@flag1({type = i1}), flag2 => @test_logical::@flag2({type = i1}), id => @test_logical::@id({type = i32}), value => @test_logical::@value({type = i32})}
            %1 = relalg.map %0 computes : [@map::@and_result({type = i32})] (%arg0: !relalg.tuple){
              %3 = relalg.getcol %arg0 @test_logical::@flag1 : i1
              %4 = relalg.getcol %arg0 @test_logical::@flag2 : i1
              %5 = db.and %3, %4 : i1, i1
              relalg.return %5 : i1
            }
            %2 = relalg.materialize %1 [@map::@and_result] => ["and_result"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a failed";
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b failed";
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, NullCoal) {
    // Test COALESCE operation with proper null flag handling
    // This mimics: SELECT COALESCE(nullable_value, 100) FROM test_table
    // Starting simple with just integer types to check null flag propagation  
    auto simpleMLIR = R"(
module {
    func.func @main() {
        // Create test table - types in basetable must be simple (i32, etc)
        %0 = relalg.basetable {table_identifier = "test_null_table|oid:12345"} columns: {
            nullable_value => @test_null_table::@nullable_value({type = i32}), 
            id => @test_null_table::@id({type = i32})
        }
        
        // Map operation with two return values to test null flags
        // First: a constant that should be NOT NULL
        // Second: the column value which may be NULL
        %1 = relalg.map %0 computes : [
            @map::@constant_result({type = i32}),
            @map::@column_value({type = i32})
        ] (%arg0: !relalg.tuple) {
            // Return a constant (should have null flag = false)
            %const_val = arith.constant 100 : i32
            
            // Get the column value (may have null flag = true)  
            %col_val = relalg.getcol %arg0 @test_null_table::@nullable_value : i32
            
            // Return both values to compare null flag handling
            relalg.return %const_val, %col_val : i32, i32
        }
        
        // Materialize both columns to see their null flags in LLVM
        %2 = relalg.materialize %1 [@map::@constant_result, @map::@column_value] => ["constant_result", "column_value"] : !dsa.table
        return
    }
}
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    // Run each phase and capture intermediate results
    std::cerr << "\n=== Testing COALESCE null flag propagation ===" << std::endl;
    
    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a (RelAlg to DB) failed";
    std::string afterPhase3a = tester->getCurrentMLIR();
    std::cerr << "After Phase 3a - checking for db.as_nullable with proper null flags..." << std::endl;
    
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b (DB to Standard) failed";  
    std::string afterPhase3b = tester->getCurrentMLIR();
    std::cerr << "After Phase 3b - checking standard MLIR representation..." << std::endl;
    
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c (Standard to LLVM) failed";
    std::string finalMLIR = tester->getCurrentMLIR();
    
    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";
    
    // Verify LLVM operations are present
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";
    
    // Log key portions for debugging null flag handling
    if (afterPhase3a.find("db.as_nullable") != std::string::npos) {
        std::cerr << "✓ db.as_nullable operations found after Phase 3a" << std::endl;
    }
    
    // Check for proper constant values (0 for false null flag)
    if (finalMLIR.find("llvm.mlir.constant(0 : i1)") != std::string::npos ||
        finalMLIR.find("llvm.mlir.constant(false)") != std::string::npos) {
        std::cerr << "✓ False null flags properly represented in LLVM IR" << std::endl;
    }
    
    std::cerr << "\n=== First 1000 chars of final LLVM IR ===" << std::endl;
    std::cerr << finalMLIR.substr(0, 1000) << std::endl;
}

TEST_F(MLIRLoweringPipelineTest, StringOps) {
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {column_order = ["id", "name", "description", "category"], table_identifier = "test_text|oid:11860936"} columns: {category => @test_text::@category({type = !db.nullable<!db.string>}), description => @test_text::@description({type = !db.nullable<!db.string>}), id => @test_text::@id({type = i32}), name => @test_text::@name({type = !db.nullable<!db.string>})}
            %1 = relalg.map %0 computes : [@map::@starts_with_a({type = i1})] (%arg0: !relalg.tuple){
              %3 = relalg.getcol %arg0 @test_text::@name : !db.nullable<!db.string>
              %4 = db.constant("A%") : !db.string
              %5 = db.runtime_call "Like"(%3, %4) : (!db.nullable<!db.string>, !db.string) -> i1
              relalg.return %5 : i1
            }
            %2 = relalg.materialize %1 [@map::@starts_with_a] => ["starts_with_a"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    // Run each phase and capture intermediate results
    std::cerr << "\n=== Testing COALESCE null flag propagation ===" << std::endl;

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a (RelAlg to DB) failed";
    std::string afterPhase3a = tester->getCurrentMLIR();
    std::cerr << "After Phase 3a - checking for db.as_nullable with proper null flags..." << std::endl;

    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b (DB to Standard) failed";
    std::string afterPhase3b = tester->getCurrentMLIR();
    std::cerr << "After Phase 3b - checking standard MLIR representation..." << std::endl;

    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c (Standard to LLVM) failed";
    std::string finalMLIR = tester->getCurrentMLIR();

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";

    std::cerr << "\n=== First 20000 chars of final LLVM IR ===" << std::endl;
    std::cerr << finalMLIR.substr(0, 20000) << std::endl;
}

TEST_F(MLIRLoweringPipelineTest, SumOp) {
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {column_order = ["id", "category", "amount", "quantity", "price"], table_identifier = "test_aggregates|oid:12721105"} columns: {amount => @test_aggregates::@amount({type = !db.nullable<i32>}), category => @test_aggregates::@category({type = !db.nullable<i32>}), id => @test_aggregates::@id({type = i32}), price => @test_aggregates::@price({type = !db.nullable<i32>}), quantity => @test_aggregates::@quantity({type = !db.nullable<i32>})}
            %1 = relalg.aggregation %0 [] computes : [@total_amount_all::@sum({type = i64})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
              %3 = relalg.aggrfn sum @test_aggregates::@amount %arg0 : !db.nullable<i32>
              relalg.return %3 : !db.nullable<i32>
            }
            %2 = relalg.materialize %1 [@total_amount_all::@sum] => ["sum"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    // Run each phase and capture intermediate results
    std::cerr << "\n=== Testing COALESCE null flag propagation ===" << std::endl;

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a (RelAlg to DB) failed";
    std::string afterPhase3a = tester->getCurrentMLIR();
    std::cerr << "After Phase 3a - checking for db.as_nullable with proper null flags..." << std::endl;

    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b (DB to Standard) failed";
    std::string afterPhase3b = tester->getCurrentMLIR();
    std::cerr << "After Phase 3b - checking standard MLIR representation..." << std::endl;

    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c (Standard to LLVM) failed";
    std::string finalMLIR = tester->getCurrentMLIR();

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";

    std::cerr << "\n=== First 20000 chars of final LLVM IR ===" << std::endl;
    std::cerr << finalMLIR.substr(0, 20000) << std::endl;
}
