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

TEST_F(MLIRLoweringPipelineTest, TestRelAlg) {
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {column_order = ["id", "value", "name"], table_identifier = "test_order_basic|oid:13892606"} columns: {id => @test_order_basic::@id({type = i32}), name => @test_order_basic::@name({type = !db.nullable<i32>}), value => @test_order_basic::@value({type = !db.nullable<i32>})}
            %1 = relalg.sort %0 [(@test_order_basic::@value,asc)]
            %2 = relalg.materialize %1 [@test_order_basic::@value,@test_order_basic::@name] => ["value", "name"] : !dsa.table
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
