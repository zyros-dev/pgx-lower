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
        pgx_lower::log::log_ir = true;
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

TEST_F(MLIRLoweringPipelineTest, TestRelAlg) {
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {column_order = ["id", "department", "employee_name", "salary", "bonus", "years_experience", "performance_score", "hire_date", "is_manager"], table_identifier = "comprehensive_data|oid:22142210"} columns: {bonus => @comprehensive_data::@bonus({type = !db.nullable<!db.decimal<8, 2>>}), department => @comprehensive_data::@department({type = !db.nullable<!db.string>}), employee_name => @comprehensive_data::@employee_name({type = !db.nullable<!db.string>}), hire_date => @comprehensive_data::@hire_date({type = !db.nullable<!db.date<day>>}), id => @comprehensive_data::@id({type = i32}), is_manager => @comprehensive_data::@is_manager({type = !db.nullable<i1>}), performance_score => @comprehensive_data::@performance_score({type = !db.nullable<!db.decimal<3, 1>>}), salary => @comprehensive_data::@salary({type = !db.nullable<!db.decimal<10, 2>>}), years_experience => @comprehensive_data::@years_experience({type = !db.nullable<i32>})}
            %1 = relalg.map %0 computes : [@map_expr::@agg_expr_0({type = !db.nullable<!db.decimal<38, 16>>})] (%arg0: !relalg.tuple){
              %5 = relalg.getcol %arg0 @comprehensive_data::@salary : !db.nullable<!db.decimal<10, 2>>
              %6 = db.constant("85000") : !db.decimal<38, 16>
              %7 = db.cast %5 : !db.nullable<!db.decimal<10, 2>> -> !db.nullable<!db.decimal<38, 16>>
              %8 = db.as_nullable %6 : !db.decimal<38, 16> -> <!db.decimal<38, 16>>
              %9 = db.sub %7 : !db.nullable<!db.decimal<38, 16>>, %8 : !db.nullable<!db.decimal<38, 16>>
              %10 = db.runtime_call "AbsDecimal"(%9) : (!db.nullable<!db.decimal<38, 16>>) -> !db.nullable<!db.decimal<38, 16>>
              relalg.return %10 : !db.nullable<!db.decimal<38, 16>>
            }
            %2 = relalg.map %1 computes : [@map_expr::@agg_expr_1({type = !db.nullable<!db.decimal<38, 16>>})] (%arg0: !relalg.tuple){
              %5 = relalg.getcol %arg0 @comprehensive_data::@bonus : !db.nullable<!db.decimal<8, 2>>
              %6 = db.constant("10000") : !db.decimal<38, 16>
              %7 = db.cast %5 : !db.nullable<!db.decimal<8, 2>> -> !db.nullable<!db.decimal<38, 16>>
              %8 = db.as_nullable %6 : !db.decimal<38, 16> -> <!db.decimal<38, 16>>
              %9 = db.sub %7 : !db.nullable<!db.decimal<38, 16>>, %8 : !db.nullable<!db.decimal<38, 16>>
              %10 = db.runtime_call "AbsDecimal"(%9) : (!db.nullable<!db.decimal<38, 16>>) -> !db.nullable<!db.decimal<38, 16>>
              relalg.return %10 : !db.nullable<!db.decimal<38, 16>>
            }
            %3 = relalg.aggregation %2 [] computes : [@aggr15::@avg_distance_from_median({type = !db.nullable<!db.decimal<38, 16>>}),@aggr15::@total_bonus_deviation({type = !db.nullable<!db.decimal<38, 16>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
              %5 = relalg.aggrfn avg @map_expr::@agg_expr_0 %arg0 : !db.nullable<!db.decimal<38, 16>>
              %6 = relalg.aggrfn sum @map_expr::@agg_expr_1 %arg0 : !db.nullable<!db.decimal<38, 16>>
              relalg.return %5, %6 : !db.nullable<!db.decimal<38, 16>>, !db.nullable<!db.decimal<38, 16>>
            }
            %4 = relalg.materialize %3 [@aggr15::@avg_distance_from_median,@aggr15::@total_bonus_deviation] => ["avg_distance_from_median", "total_bonus_deviation"] : !dsa.table
            return
          }
        }
        )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

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
