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
            %0 = relalg.basetable  {column_order = ["bool_col", "int2_col", "int4_col", "int8_col", "float4_col", "float8_col", "string_col", "char_col", "text_col", "decimal_col", "numeric_col", "date_col", "timestamp_col", "interval_col"], table_identifier = "type_test_table|oid:19528750"} columns: {bool_col => @type_test_table::@bool_col({type = !db.nullable<i1>}), char_col => @type_test_table::@char_col({type = !db.nullable<!db.string>}), date_col => @type_test_table::@date_col({type = !db.nullable<!db.date<day>>}), decimal_col => @type_test_table::@decimal_col({type = !db.nullable<!db.decimal<10, 2>>}), float4_col => @type_test_table::@float4_col({type = !db.nullable<f32>}), float8_col => @type_test_table::@float8_col({type = !db.nullable<f64>}), int2_col => @type_test_table::@int2_col({type = !db.nullable<i16>}), int4_col => @type_test_table::@int4_col({type = !db.nullable<i32>}), int8_col => @type_test_table::@int8_col({type = !db.nullable<i64>}), interval_col => @type_test_table::@interval_col({type = !db.nullable<!db.interval<daytime>>}), numeric_col => @type_test_table::@numeric_col({type = !db.nullable<!db.decimal<15, 5>>}), string_col => @type_test_table::@string_col({type = !db.nullable<!db.string>}), text_col => @type_test_table::@text_col({type = !db.nullable<!db.string>}), timestamp_col => @type_test_table::@timestamp_col({type = !db.nullable<!db.timestamp<microsecond>>})}
            %1 = relalg.aggregation %0 [] computes : [@aggr0::@sum_0({type = !db.nullable<!db.decimal<21, 16>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
              %3 = relalg.aggrfn sum @type_test_table::@int8_col %arg0 : !db.nullable<!db.decimal<21, 16>>
              relalg.return %3 : !db.nullable<!db.decimal<21, 16>>
            }
            %2 = relalg.materialize %1 [@aggr0::@sum_0] => ["sum_0"] : !dsa.table
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
