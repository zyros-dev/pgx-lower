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
          func.func @main() -> !dsa.table {
            %0 = relalg.basetable  {column_order = ["dept_id", "dept_name", "location"], table_identifier = "departments|oid:22617391"} columns: {dept_id => @d::@dept_id({type = !db.nullable<i32>}), dept_name => @d::@dept_name({type = !db.nullable<!db.string>}), location => @d::@location({type = !db.nullable<!db.string>})}
            %1 = relalg.sort %0 [(@d::@dept_id,asc)]
            %2 = relalg.basetable  {column_order = ["emp_id", "emp_name", "dept_id", "salary"], table_identifier = "employees|oid:22617386"} columns: {dept_id => @e::@dept_id({type = !db.nullable<i32>}), emp_id => @e::@emp_id({type = !db.nullable<i32>}), emp_name => @e::@emp_name({type = !db.nullable<!db.string>}), salary => @e::@salary({type = !db.nullable<i32>})}
            %3 = relalg.sort %2 [(@e::@dept_id,asc)]
            %4 = relalg.outerjoin %1, %3 (%arg0: !relalg.tuple){
              %true = arith.constant true
              relalg.return %true : i1
            }  mapping: {@e::@emp_name_nullable({type = !db.nullable<!db.string>})=[@e::@emp_name], @e::@salary_nullable({type = !db.nullable<i32>})=[@e::@salary], @e::@emp_id_nullable({type = !db.nullable<i32>})=[@e::@emp_id], @e::@dept_id_nullable({type = !db.nullable<i32>})=[@e::@dept_id]}
            %5 = relalg.projection all [@e::@emp_name_nullable,@e::@salary_nullable,@d::@dept_name,@d::@location] %4
            %6 = relalg.materialize %5 [@e::@emp_name_nullable,@e::@salary_nullable,@d::@dept_name,@d::@location] => ["emp_name", "salary", "dept_name", "location"] : !dsa.table
            return %6 : !dsa.table
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
