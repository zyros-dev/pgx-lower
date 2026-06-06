extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder{&ctx};

    Fixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::dsa::DSADialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto moduleToString(mlir::ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    return output;
}

auto runDBToStd(mlir::MLIRContext& ctx, mlir::ModuleOp module) -> bool {
    mlir::PassManager pm(&ctx);
    mlir::db::createLowerDBPipeline(pm);
    return mlir::succeeded(pm.run(module));
}

void requireContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) == std::string::npos) {
        elog(ERROR, "%s:%d expected to find '%s'", __FILE__, __LINE__, needle.str().c_str());
    }
}

void requireNotContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) != std::string::npos) {
        elog(ERROR, "%s:%d unexpected '%s' present", __FILE__, __LINE__, needle.str().c_str());
    }
}

void requireIntArrayAttr(mlir::Operation* op, llvm::StringRef name, llvm::ArrayRef<int64_t> expected) {
    const auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
    REQUIRE(attr);
    REQUIRE(attr.size() == expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto value = mlir::cast<mlir::IntegerAttr>(attr[index]).getInt();
        if (value != expected[index]) {
            elog(ERROR, "%s:%d expected '%s'[%zu] = %ld got %ld", __FILE__, __LINE__, name.str().c_str(), index,
                 expected[index], value);
        }
    }
}

} // namespace

PGX_TEST_FN(pg_physicalization_converts_pg_primitive_carriers) {
    Fixture f;
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @carriers(%i4: !db.pg_int4,
                      %nullable_i4: !db.pg_int4<nullable>,
                      %numeric: !db.pg_numeric<typmod = -1>,
                      %text: !db.pg_text<collation = 100>) {
    return
  }
}
)mlir",
                                                          &f.ctx);
    REQUIRE(module);
    REQUIRE(runDBToStd(f.ctx, *module));

    const auto printed = moduleToString(*module);
    requireContains(printed, "i32");
    requireContains(printed, "tuple<i1, i32>");
    requireContains(printed, "i64");
    requireContains(printed, "!util.varlen32");
    requireNotContains(printed, "!db.pg_");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_physicalization_records_dsa_oid_typmod_collation_attrs) {
    Fixture f;
    auto module = mlir::ModuleOp::create(f.builder.getUnknownLoc());
    f.builder.setInsertionPointToStart(module.getBody());
    auto func = f.builder.create<mlir::func::FuncOp>(f.builder.getUnknownLoc(), "metadata",
                                                     f.builder.getFunctionType({}, {}));
    auto* entry = func.addEntryBlock();
    f.builder.setInsertionPointToStart(entry);

    auto varchar = mlir::db::PgVarcharType::get(&f.ctx, 14, 777);
    auto numeric = mlir::db::PgNumericType::get(&f.ctx, 786438);
    auto text = mlir::db::PgTextType::get(&f.ctx, 777);
    auto keyTuple = mlir::TupleType::get(&f.ctx, {varchar, numeric});
    auto valTuple = mlir::TupleType::get(&f.ctx, {text});
    auto sortKeys = f.builder.getArrayAttr(
        {f.builder.getArrayAttr({f.builder.getI32IntegerAttr(0), f.builder.getI32IntegerAttr(0)})});

    f.builder.create<mlir::dsa::CreateDS>(
        f.builder.getUnknownLoc(), mlir::dsa::GenericIterableType::get(&f.ctx, keyTuple, "pgsort_iterator"), sortKeys);
    f.builder.create<mlir::dsa::CreateDS>(f.builder.getUnknownLoc(),
                                          mlir::dsa::JoinHashtableType::get(&f.ctx, keyTuple, valTuple));
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc());

    REQUIRE(runDBToStd(f.ctx, module));

    mlir::dsa::CreateDS sortCreate;
    mlir::dsa::CreateDS joinCreate;
    module.walk([&](mlir::dsa::CreateDS op) {
        if (const auto generic = mlir::dyn_cast<mlir::dsa::GenericIterableType>(op.getDs().getType());
            generic && generic.getIteratorName() == "pgsort_iterator")
        {
            sortCreate = op;
        } else if (mlir::isa<mlir::dsa::JoinHashtableType>(op.getDs().getType())) {
            joinCreate = op;
        }
    });
    REQUIRE(sortCreate);
    REQUIRE(joinCreate);

    requireIntArrayAttr(sortCreate, "pgx_original_type_oids", {VARCHAROID, NUMERICOID});
    requireIntArrayAttr(sortCreate, "pgx_original_type_typmods", {14, 786438});
    requireIntArrayAttr(sortCreate, "pgx_original_type_collations", {777, InvalidOid});
    requireIntArrayAttr(joinCreate, "pgx_original_key_type_oids", {VARCHAROID, NUMERICOID});
    requireIntArrayAttr(joinCreate, "pgx_original_key_type_typmods", {14, 786438});
    requireIntArrayAttr(joinCreate, "pgx_original_key_type_collations", {777, InvalidOid});
    requireIntArrayAttr(joinCreate, "pgx_original_val_type_oids", {TEXTOID});
    requireIntArrayAttr(joinCreate, "pgx_original_val_type_typmods", {-1});
    requireIntArrayAttr(joinCreate, "pgx_original_val_type_collations", {777});
    PG_RETURN_VOID();
}
