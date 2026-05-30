extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "pgx-lower/test/standalone_mlir_runner.h"
#include "pgx-lower/test/pgx_test_fn.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <memory>
#include <string>

#define REQUIRE(cond) \
    do { if (!(cond)) elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); } while (0)

PGX_TEST_FN(pipeline_mapop_print) {
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();
    auto* builder = tester->getBuilder();
    auto& column_manager = tester->getColumnManager();

    const char* const base_table_mlir = R"(
        module {
          func.func @main() -> !dsa.table {
            %0 = relalg.basetable  {column_order = ["id"], table_identifier = "test|oid:32970940"} columns: {id => @test::@id({type = i32})}
            %1 = relalg.materialize %0 [@test::@id] => ["id"] : !dsa.table
            return %1 : !dsa.table
          }
        }
    )";

    REQUIRE(tester->loadRelAlgModule(base_table_mlir));

    auto module = tester->getModule();
    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    REQUIRE(main_func);

    auto& entry_block = main_func.getBody().front();
    auto& base_table_op = *entry_block.begin();

    builder->setInsertionPoint(&entry_block, ++entry_block.begin());

    auto col_def = column_manager.createDef("maptest", "computed");
    col_def.getColumn().type = builder->getI32Type();

    auto map_op = builder->create<mlir::relalg::MapOp>(
        builder->getUnknownLoc(),
        base_table_op.getResult(0),
        builder->getArrayAttr({col_def})
    );
    (void) map_op;

    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    stream.flush();

    REQUIRE(!output.empty());
    PG_RETURN_VOID();
}
