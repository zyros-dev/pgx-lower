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
            %0 = relalg.basetable  {column_order = ["id", "product_name", "category", "order_amount", "quantity", "customer_type", "order_date", "region"], table_identifier = "product_orders|oid:20929595"} columns: {category => @product_orders::@category({type = !db.nullable<!db.string>}), customer_type => @product_orders::@customer_type({type = !db.nullable<!db.string>}), id => @product_orders::@id({type = i32}), order_amount => @product_orders::@order_amount({type = !db.nullable<!db.decimal<10, 2>>}), order_date => @product_orders::@order_date({type = !db.nullable<!db.date<day>>}), product_name => @product_orders::@product_name({type = !db.nullable<!db.string>}), quantity => @product_orders::@quantity({type = !db.nullable<i32>}), region => @product_orders::@region({type = !db.nullable<!db.string>})}
            %1 = relalg.selection %0 (%arg0: !relalg.tuple){
              %5 = relalg.getcol %arg0 @product_orders::@customer_type : !db.nullable<!db.string>
              %6 = db.constant("Premium") : !db.string
              %7 = db.as_nullable %6 : !db.string -> <!db.string>
              %8 = db.as_nullable %6 : !db.string -> <!db.string>
              %9 = db.compare eq %5 : !db.nullable<!db.string>, %8 : !db.nullable<!db.string>
              %10 = db.derive_truth %9 : !db.nullable<i1>
              relalg.return %10 : i1
            }
            %2 = relalg.sort %1 [(@product_orders::@region,asc)]
            %3 = relalg.aggregation %2 [@product_orders::@id] computes : [@aggr1::@sum_0({type = !db.nullable<!db.decimal<38, 16>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
              %5 = relalg.aggrfn sum @product_orders::@order_amount %arg0 : !db.nullable<!db.decimal<38, 16>>
              relalg.return %5 : !db.nullable<!db.decimal<38, 16>>
            }
            %4 = relalg.materialize %3 [@product_orders::@id,@aggr1::@sum_0] => ["id", "sum_0"] : !dsa.table
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
