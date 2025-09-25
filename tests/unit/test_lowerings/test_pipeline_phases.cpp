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
            %0 = relalg.basetable  {column_order = ["dept_id", "dept_name", "location"], table_identifier = "departments|oid:22653544"} columns: {dept_id => @d::@dept_id({type = !db.nullable<i32>}), dept_name => @d::@dept_name({type = !db.nullable<!db.string>}), location => @d::@location({type = !db.nullable<!db.string>})}
            %1 = relalg.sort %0 [(@d::@dept_id,asc)]
            %2 = relalg.basetable  {column_order = ["emp_id", "emp_name", "dept_id", "salary"], table_identifier = "employees|oid:22653539"} columns: {dept_id => @e::@dept_id({type = !db.nullable<i32>}), emp_id => @e::@emp_id({type = !db.nullable<i32>}), emp_name => @e::@emp_name({type = !db.nullable<!db.string>}), salary => @e::@salary({type = !db.nullable<i32>})}
            %3 = relalg.sort %2 [(@e::@dept_id,asc)]
            %4 = relalg.outerjoin %3, %1 (%arg0: !relalg.tuple){
              %8 = relalg.getcol %arg0 @d::@dept_id : !db.nullable<i32>
              %9 = relalg.getcol %arg0 @e::@dept_id : !db.nullable<i32>
              %10 = db.compare eq %8 : !db.nullable<i32>, %9 : !db.nullable<i32>
              %11 = db.derive_truth %10 : !db.nullable<i1>
              relalg.return %11 : i1
            }  mapping: {@oj0::@dept_name({type = !db.nullable<!db.string>})=[@d::@dept_name], @oj0::@location({type = !db.nullable<!db.string>})=[@d::@location], @oj0::@dept_id({type = !db.nullable<i32>})=[@d::@dept_id]}
            %5 = relalg.selection %4 (%arg0: !relalg.tuple){
              %8 = relalg.getcol %arg0 @d::@location : !db.nullable<!db.string>
              %9 = db.constant("Building A") : !db.string
              %10 = db.as_nullable %9 : !db.string -> <!db.string>
              %11 = db.as_nullable %9 : !db.string -> <!db.string>
              %12 = db.compare eq %8 : !db.nullable<!db.string>, %11 : !db.nullable<!db.string>
              %13 = db.derive_truth %12 : !db.nullable<i1>
              %14 = relalg.getcol %arg0 @employees::@emp_name : !db.nullable<!db.string>
              %15 = db.isnull %14 : <!db.string>
              %16 = db.or %13, %15 : i1, i1
              relalg.return %16 : i1
            }
            %6 = relalg.projection all [@e::@emp_name,@oj0::@dept_name] %5
            %7 = relalg.materialize %6 [@e::@emp_name,@oj0::@dept_name] => ["emp_name", "dept_name"] : !dsa.table
            return %7 : !dsa.table
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
