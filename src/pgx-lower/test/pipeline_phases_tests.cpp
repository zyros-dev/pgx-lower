extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "pgx-lower/test/standalone_mlir_runner.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <memory>
#include <string>

#define REQUIRE(cond) \
    do { if (!(cond)) elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); } while (0)

extern "C" {

PG_FUNCTION_INFO_V1(ts_test_pipeline_mapop_print);
Datum ts_test_pipeline_mapop_print(PG_FUNCTION_ARGS) {
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();
    auto* builder = tester->getBuilder();
    auto& columnManager = tester->getColumnManager();

    const char* baseTableMLIR = R"(
        module {
          func.func @main() -> !dsa.table {
            %0 = relalg.basetable  {column_order = ["id"], table_identifier = "test|oid:32970940"} columns: {id => @test::@id({type = i32})}
            %1 = relalg.materialize %0 [@test::@id] => ["id"] : !dsa.table
            return %1 : !dsa.table
          }
        }
    )";

    REQUIRE(tester->loadRelAlgModule(baseTableMLIR));

    auto module = tester->getModule();
    auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
    REQUIRE(mainFunc);

    auto& entryBlock = mainFunc.getBody().front();
    auto& baseTableOp = *entryBlock.begin();

    builder->setInsertionPoint(&entryBlock, ++entryBlock.begin());

    auto colDef = columnManager.createDef("maptest", "computed");
    colDef.getColumn().type = builder->getI32Type();

    auto mapOp = builder->create<mlir::relalg::MapOp>(
        builder->getUnknownLoc(),
        baseTableOp.getResult(0),
        builder->getArrayAttr({colDef})
    );
    (void) mapOp;

    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    stream.flush();

    REQUIRE(!output.empty());
    PG_RETURN_VOID();
}

}  // extern "C"
