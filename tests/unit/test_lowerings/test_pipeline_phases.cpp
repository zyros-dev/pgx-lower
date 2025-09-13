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
            %0 = relalg.basetable  {column_order = ["id", "department", "product", "amount", "quantity", "sales_date"], table_identifier = "sales_data|oid:15694888"} columns: {amount => @sales_data::@amount({type = !db.nullable<!db.decimal<10, 2>>}), department => @sales_data::@department({type = !db.nullable<!db.string>}), id => @sales_data::@id({type = i32}), product => @sales_data::@product({type = !db.nullable<!db.string>}), quantity => @sales_data::@quantity({type = !db.nullable<i32>}), sales_date => @sales_data::@sales_date({type = !db.nullable<!db.date<day>>})}
            %1 = relalg.aggregation %0 [@sales_data::@department] computes : [@total_sales::@sum({type = !db.nullable<!db.decimal<10, 2>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
              %4 = relalg.aggrfn sum @sales_data::@amount %arg0 : !db.nullable<!db.decimal<10, 2>>
              relalg.return %4 : !db.nullable<!db.decimal<10, 2>>
            }
            %2 = relalg.sort %1 [(@sales_data::@department,asc)]
            %3 = relalg.materialize %2 [@sales_data::@department,@total_sales::@sum] => ["department", "total_sales"] : !dsa.table
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
