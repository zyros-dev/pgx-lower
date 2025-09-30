#include <gtest/gtest.h>
#include "standalone_mlir_runner.h"
#include "pgx-lower/utility/logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
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

TEST_F(MLIRLoweringPipelineTest, MapOpPrintCrashTest) {
    // This test reproduces the crash that occurs when printing MapOp with ColumnDefAttr
    // We build the MapOp programmatically to avoid parser issues

    std::cerr << "\n=== Testing MapOp printing crash ===" << std::endl;

    auto context = tester->getContext();
    auto builder = tester->getBuilder();
    auto& columnManager = tester->getColumnManager();

    // Create a simple BaseTableOp first
    std::cerr << "Creating BaseTableOp..." << std::endl;
    auto baseTableMLIR = R"(
        module {
          func.func @main() -> !relalg.tuplestream {
            %0 = relalg.basetable {column_order = ["id"], table_identifier = "test|oid:123"} columns: {
              id => @test::@id({type = i32})
            }
            return %0 : !relalg.tuplestream
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(baseTableMLIR)) << "Failed to load base module";
    std::cerr << "Base module loaded successfully" << std::endl;

    // Now programmatically build a MapOp with ColumnDefAttr
    std::cerr << "Building MapOp programmatically..." << std::endl;

    auto module = tester->getModule();
    auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(mainFunc) << "Failed to find main function";

    auto& entryBlock = mainFunc.getBody().front();
    auto& baseTableOp = *entryBlock.begin();

    // Create MapOp
    builder->setInsertionPoint(&entryBlock, ++entryBlock.begin());

    auto colDef = columnManager.createDef("maptest", "computed");
    colDef.getColumn().type = builder->getI32Type();

    std::cerr << "Creating MapOp with ColumnDefAttr..." << std::endl;
    auto mapOp = builder->create<mlir::relalg::MapOp>(
        builder->getUnknownLoc(),
        baseTableOp.getResult(0),
        builder->getArrayAttr({colDef})
    );

    std::cerr << "MapOp created, now trying to print it..." << std::endl;

    // Try to print the module - this should crash
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    stream.flush();

    std::cerr << "Successfully printed MapOp! Output:" << std::endl;
    std::cerr << output << std::endl;

    SUCCEED() << "If we get here, printing worked!";
}
