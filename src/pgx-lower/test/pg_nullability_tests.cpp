extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

constexpr auto kDefaultCollation = 100;
constexpr auto kTypmodUnconstrained = -1;

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder{&ctx};

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::dsa::DSADialect>();
        ctx.loadDialect<mlir::arith::ArithDialect>();
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

auto countSubstring(const std::string& haystack, llvm::StringRef needle) -> size_t {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle.str(), pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

auto parseModule(mlir::MLIRContext& ctx, llvm::StringRef moduleText, bool logDiagnostics = true)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(moduleText), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler handler(sourceMgr, &ctx, stream);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
    if (!module && logDiagnostics && !diagnostics.empty()) {
        elog(WARNING, "MLIR parse failed: %s", diagnostics.c_str());
    }
    return module;
}

auto runDBToStd(mlir::MLIRContext& ctx, mlir::ModuleOp module) -> bool {
    mlir::PassManager pm(&ctx);
    mlir::db::createLowerDBPipeline(pm);
    return mlir::succeeded(pm.run(module)) && mlir::succeeded(mlir::verify(module));
}

auto runDSAToStd(mlir::MLIRContext& ctx, mlir::ModuleOp module) -> bool {
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::dsa::createLowerToStdPass());
    return mlir::succeeded(pm.run(module)) && mlir::succeeded(mlir::verify(module));
}

} // namespace

PGX_TEST_FN(pg_nullability_combinators) {
    Fixture f;

    mlir::Type never = mlir::db::PgInt4Type::get(&f.ctx);
    mlir::Type maybe = mlir::db::PgInt4Type::get(&f.ctx, mlir::db::PgNullability::Maybe);

    llvm::SmallVector<mlir::Type> neverNever{never, never};
    REQUIRE(mlir::db::combineSqlNullability(neverNever) == mlir::db::PgNullability::Never);

    llvm::SmallVector<mlir::Type> neverMaybe{never, maybe};
    REQUIRE(mlir::db::combineSqlNullability(neverMaybe) == mlir::db::PgNullability::Maybe);

    llvm::SmallVector<mlir::Type> maybeMaybe{maybe, maybe};
    REQUIRE(mlir::db::combineSqlNullability(maybeMaybe) == mlir::db::PgNullability::Maybe);

    mlir::Type legacyNullable = mlir::db::NullableType::get(&f.ctx, never);
    llvm::SmallVector<mlir::Type> pgLegacyMixed{never, legacyNullable};
    REQUIRE(mlir::db::combineSqlNullability(pgLegacyMixed) == mlir::db::PgNullability::Maybe);

    auto module = mlir::ModuleOp::create(f.builder.getUnknownLoc());
    f.builder.setInsertionPointToStart(module.getBody());
    auto fn = f.builder.create<mlir::func::FuncOp>(f.builder.getUnknownLoc(), "combine_values",
                                                   f.builder.getFunctionType({never, maybe}, {}));
    auto* block = fn.addEntryBlock();
    llvm::SmallVector<mlir::Value> values{block->getArgument(0), block->getArgument(1)};
    REQUIRE(mlir::db::combineSqlNullability(mlir::ValueRange(values)) == mlir::db::PgNullability::Maybe);

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_db_ops_roundtrip_preserves_explicit_pg_result_types) {
    Fixture f;

    const char* moduleText = R"mlir(
module {
  func.func @ops(%a: !db.pg_numeric<typmod = -1>,
                 %b: !db.pg_numeric<typmod = -1, nullable>,
                 %c: !db.pg_int8,
                 %d: !db.pg_int8) {
    %sum = db.add %a : !db.pg_numeric<typmod = -1>, %b : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_numeric<typmod = -1, nullable>
    %cmp = db.compare lt %a : !db.pg_numeric<typmod = -1>, %b : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_bool<nullable>
    %same_carrier_sum = db.add %c : !db.pg_int8, %d : !db.pg_int8 -> !db.pg_numeric<typmod = -1>
    %is_null = db.isnull %b : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_bool
    %cast = db.cast %b : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_text<collation = 100, nullable>
    return
  }
}
)mlir";

    auto module = parseModule(f.ctx, moduleText);
    REQUIRE(module);
    const std::string printed = moduleToString(*module);
    requireContains(printed, "db.add");
    requireContains(printed, "-> !db.pg_numeric<typmod = -1, nullable>");
    requireContains(printed, "db.compare");
    requireContains(printed, "-> !db.pg_bool<nullable>");
    requireContains(printed, "-> !db.pg_numeric<typmod = -1>");
    requireContains(printed, "db.isnull");
    requireContains(printed, "-> !db.pg_bool");
    requireContains(printed, "db.cast");
    requireContains(printed, "-> !db.pg_text<collation = 100, nullable>");

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_db_ops_reject_invalid_pg_result_types) {
    Fixture f;

    auto missingResultType = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_numeric<typmod = -1>, %b: !db.pg_numeric<typmod = -1>) {
    %sum = db.add %a : !db.pg_numeric<typmod = -1>, %b : !db.pg_numeric<typmod = -1>
    return
  }
}
)mlir",
                                         false);
    REQUIRE(!missingResultType);

    auto nonPgArithmeticResult = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_numeric<typmod = -1>, %b: !db.pg_numeric<typmod = -1>) {
    %sum = db.add %a : !db.pg_numeric<typmod = -1>, %b : !db.pg_numeric<typmod = -1> -> i64
    return
  }
}
)mlir",
                                             false);
    REQUIRE(!nonPgArithmeticResult);

    auto nonBoolComparisonResult = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_numeric<typmod = -1>, %b: !db.pg_numeric<typmod = -1>) {
    %cmp = db.compare lt %a : !db.pg_numeric<typmod = -1>, %b : !db.pg_numeric<typmod = -1> -> !db.pg_int4
    return
  }
}
)mlir",
                                               false);
    REQUIRE(!nonBoolComparisonResult);

    auto nullableIsNullResult = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_int4<nullable>) {
    %is_null = db.isnull %a : !db.pg_int4<nullable> -> !db.pg_bool<nullable>
    return
  }
}
)mlir",
                                            false);
    REQUIRE(!nullableIsNullResult);

    auto nonPgCastResult = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_int4) {
    %cast = db.cast %a : !db.pg_int4 -> i64
    return
  }
}
)mlir",
                                       false);
    REQUIRE(!nonPgCastResult);

    auto wrongCastNullability = parseModule(f.ctx, R"mlir(
module {
  func.func @bad(%a: !db.pg_int4<nullable>) {
    %cast = db.cast %a : !db.pg_int4<nullable> -> !db.pg_int8
    return
  }
}
)mlir",
                                            false);
    REQUIRE(!wrongCastNullability);

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_constant_and_null_pg_type_contract) {
    Fixture f;

    auto good = parseModule(f.ctx, R"mlir(
module {
  func.func @constants() {
    %i = db.constant(42 : i32) : !db.pg_int4
    %n = db.constant(12345 : i64) : !db.pg_numeric<typmod = -1>
    %s = db.constant("abc") : !db.pg_text<collation = 100>
    %null_i = db.null : !db.pg_int4<nullable>
    %null_s = db.null : !db.pg_text<collation = 100, nullable>
    return
  }
}
)mlir");
    REQUIRE(good);
    const std::string printed = moduleToString(*good);
    requireContains(printed, "db.constant(42 : i32) : !db.pg_int4");
    requireContains(printed, "db.null : !db.pg_int4<nullable>");
    requireContains(printed, "db.null : !db.pg_text<collation = 100, nullable>");

    auto nullablePgNumericToFloatCast = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_numeric_to_float(%v: !db.pg_numeric<typmod = -1, nullable>) -> !db.pg_float8<nullable> {
    %cast = db.cast %v : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_float8<nullable>
    return %cast : !db.pg_float8<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgNumericToFloatCast);
    REQUIRE(runDBToStd(f.ctx, *nullablePgNumericToFloatCast));
    const std::string nullablePgNumericToFloatCastLowered = moduleToString(*nullablePgNumericToFloatCast);
    requireNotContains(nullablePgNumericToFloatCastLowered, "db.cast");
    requireContains(nullablePgNumericToFloatCastLowered, "arith.select");
    requireContains(nullablePgNumericToFloatCastLowered, "pgx_numeric_to_float");

    auto nullablePgNumericCompare = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_numeric_compare(%left: !db.pg_numeric<typmod = 786438>, %right: !db.pg_numeric<typmod = -1, nullable>) -> !db.pg_bool<nullable> {
    %cast = db.cast %right : !db.pg_numeric<typmod = -1, nullable> -> !db.pg_numeric<typmod = 786438, nullable>
    %cmp = db.compare lt %left : !db.pg_numeric<typmod = 786438>, %cast : !db.pg_numeric<typmod = 786438, nullable> -> !db.pg_bool<nullable>
    return %cmp : !db.pg_bool<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgNumericCompare);
    REQUIRE(runDBToStd(f.ctx, *nullablePgNumericCompare));
    requireNotContains(moduleToString(*nullablePgNumericCompare), "db.compare");

    auto nullablePgKeyHash = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_key_hash(%v: !db.pg_int4) -> index {
    %nullable = db.as_nullable %v : !db.pg_int4 -> !db.pg_int4<nullable>
    %key = util.pack %nullable : !db.pg_int4<nullable> -> tuple<!db.pg_int4<nullable>>
    %hash = db.hash %key : tuple<!db.pg_int4<nullable>>
    return %hash : index
  }
}
)mlir");
    REQUIRE(nullablePgKeyHash);
    REQUIRE(runDBToStd(f.ctx, *nullablePgKeyHash));
    requireNotContains(moduleToString(*nullablePgKeyHash), "db.hash");
    requireContains(moduleToString(*nullablePgKeyHash), "util.hash_64");

    auto nullablePgIntCompare = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_int_compare(%left: !db.pg_int4, %right: !db.pg_int4) -> !db.pg_bool<nullable> {
    %nullable = db.as_nullable %right : !db.pg_int4 -> !db.pg_int4<nullable>
    %cmp = db.compare eq %left : !db.pg_int4, %nullable : !db.pg_int4<nullable> -> !db.pg_bool<nullable>
    return %cmp : !db.pg_bool<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgIntCompare);
    REQUIRE(runDBToStd(f.ctx, *nullablePgIntCompare));
    requireNotContains(moduleToString(*nullablePgIntCompare), "db.compare");

    auto nullablePgIntCast = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_int_cast(%value: !db.pg_int4<nullable>) -> !db.pg_int8<nullable> {
    %cast = db.cast %value : !db.pg_int4<nullable> -> !db.pg_int8<nullable>
    return %cast : !db.pg_int8<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgIntCast);
    REQUIRE(runDBToStd(f.ctx, *nullablePgIntCast));
    const std::string nullablePgIntCastLowered = moduleToString(*nullablePgIntCast);
    requireNotContains(nullablePgIntCastLowered, "db.cast");
    requireContains(nullablePgIntCastLowered, "arith.extsi");

    auto pgStringCast = parseModule(f.ctx, R"mlir(
module {
  func.func @pg_string_cast(%value: !db.pg_text<collation = 100>) -> !db.pg_varchar<typmod = 24, collation = 100> {
    %cast = db.cast %value : !db.pg_text<collation = 100> -> !db.pg_varchar<typmod = 24, collation = 100>
    return %cast : !db.pg_varchar<typmod = 24, collation = 100>
  }
}
)mlir");
    REQUIRE(pgStringCast);
    REQUIRE(runDBToStd(f.ctx, *pgStringCast));
    requireNotContains(moduleToString(*pgStringCast), "db.cast");

    auto nullablePgStringCompare = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_string_compare(%left: !db.pg_varchar<typmod = 24, collation = 100, nullable>, %right: !db.pg_varchar<typmod = 24, collation = 100>) -> !db.pg_bool<nullable> {
    %cmp = db.compare eq %left : !db.pg_varchar<typmod = 24, collation = 100, nullable>, %right : !db.pg_varchar<typmod = 24, collation = 100> -> !db.pg_bool<nullable>
    return %cmp : !db.pg_bool<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgStringCompare);
    REQUIRE(runDBToStd(f.ctx, *nullablePgStringCompare));
    const std::string nullablePgStringCompareLowered = moduleToString(*nullablePgStringCompare);
    requireNotContains(nullablePgStringCompareLowered, "db.compare");
    requireNotContains(nullablePgStringCompareLowered, "(tuple<i1, !util.varlen32>, !util.varlen32)");

    auto nullablePgMixedIntArithmetic = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_mixed_int_arithmetic(%left: !db.pg_int2<nullable>, %right: !db.pg_int4<nullable>) -> !db.pg_int4<nullable> {
    %cast = db.cast %right : !db.pg_int4<nullable> -> !db.pg_int2<nullable>
    %sum = db.add %left : !db.pg_int2<nullable>, %cast : !db.pg_int2<nullable> -> !db.pg_int4<nullable>
    return %sum : !db.pg_int4<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgMixedIntArithmetic);
    REQUIRE(runDBToStd(f.ctx, *nullablePgMixedIntArithmetic));
    const std::string nullablePgMixedIntArithmeticLowered = moduleToString(*nullablePgMixedIntArithmetic);
    requireNotContains(nullablePgMixedIntArithmeticLowered, "db.add");
    requireContains(nullablePgMixedIntArithmeticLowered, "arith.extsi");

    auto nullablePgIntArithmetic = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_int_arithmetic(%left: !db.pg_int4<nullable>, %right: !db.pg_int4<nullable>) -> (!db.pg_int4<nullable>, !db.pg_int4<nullable>, !db.pg_int4<nullable>, !db.pg_int4<nullable>) {
    %add = db.add %left : !db.pg_int4<nullable>, %right : !db.pg_int4<nullable> -> !db.pg_int4<nullable>
    %sub = db.sub %left : !db.pg_int4<nullable>, %right : !db.pg_int4<nullable> -> !db.pg_int4<nullable>
    %mul = db.mul %left : !db.pg_int4<nullable>, %right : !db.pg_int4<nullable> -> !db.pg_int4<nullable>
    %div = db.div %left : !db.pg_int4<nullable>, %right : !db.pg_int4<nullable> -> !db.pg_int4<nullable>
    return %add, %sub, %mul, %div : !db.pg_int4<nullable>, !db.pg_int4<nullable>, !db.pg_int4<nullable>, !db.pg_int4<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgIntArithmetic);
    REQUIRE(runDBToStd(f.ctx, *nullablePgIntArithmetic));
    const std::string nullablePgIntArithmeticLowered = moduleToString(*nullablePgIntArithmetic);
    requireNotContains(nullablePgIntArithmeticLowered, "db.add");
    requireNotContains(nullablePgIntArithmeticLowered, "db.sub");
    requireNotContains(nullablePgIntArithmeticLowered, "db.mul");
    requireNotContains(nullablePgIntArithmeticLowered, "db.div");
    requireNotContains(nullablePgIntArithmeticLowered, "(tuple<i1, i32>, tuple<i1, i32>)");

    auto nullablePgBoolLogic = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_bool_logic(%left: !db.pg_bool<nullable>, %right: !db.pg_bool<nullable>) -> (!db.pg_bool<nullable>, !db.pg_bool<nullable>, !db.pg_bool<nullable>) {
    %or = db.or %left, %right : !db.pg_bool<nullable>, !db.pg_bool<nullable> -> !db.pg_bool<nullable>
    %and = db.and %left, %right : !db.pg_bool<nullable>, !db.pg_bool<nullable> -> !db.pg_bool<nullable>
    %not = db.not %left : !db.pg_bool<nullable> -> !db.pg_bool<nullable>
    return %or, %and, %not : !db.pg_bool<nullable>, !db.pg_bool<nullable>, !db.pg_bool<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgBoolLogic);
    REQUIRE(runDBToStd(f.ctx, *nullablePgBoolLogic));
    const std::string nullablePgBoolLogicLowered = moduleToString(*nullablePgBoolLogic);
    requireNotContains(nullablePgBoolLogicLowered, "db.or");
    requireNotContains(nullablePgBoolLogicLowered, "db.and");
    requireNotContains(nullablePgBoolLogicLowered, "db.not");
    requireNotContains(nullablePgBoolLogicLowered, "(tuple<i1, i1>, tuple<i1, i1>)");

    auto nullablePgTextLike = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_pg_text_like(%value: !db.pg_text<collation = 100, nullable>, %pattern: !db.pg_text<collation = 100>) -> !db.pg_bool<nullable> {
    %like = db.runtime_call "Like"(%value, %pattern) : (!db.pg_text<collation = 100, nullable>, !db.pg_text<collation = 100>) -> !db.pg_bool<nullable>
    return %like : !db.pg_bool<nullable>
  }
}
)mlir");
    REQUIRE(nullablePgTextLike);
    REQUIRE(runDBToStd(f.ctx, *nullablePgTextLike));
    const std::string nullablePgTextLikeLowered = moduleToString(*nullablePgTextLike);
    requireNotContains(nullablePgTextLikeLowered, "db.runtime_call");
    requireNotContains(nullablePgTextLikeLowered, "(tuple<i1, !util.varlen32>, !util.varlen32)");

    auto nullableRecordAt = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_record_at(%record: !dsa.record<tuple<tuple<i1, i32>, i32>>) -> tuple<i1, i32> {
    %value = dsa.at %record[0] : <tuple<tuple<i1, i32>, i32>> -> tuple<i1, i32>
    return %value : tuple<i1, i32>
  }
}
)mlir");
    REQUIRE(nullableRecordAt);
    REQUIRE(runDSAToStd(f.ctx, *nullableRecordAt));
    const std::string nullableRecordAtLowered = moduleToString(*nullableRecordAt);
    requireNotContains(nullableRecordAtLowered, "dsa.at");
    requireContains(nullableRecordAtLowered, "util.pack");

    auto nullableTableBuilderAppend = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_table_builder_append(%is_null: i1, %payload: i32) -> !dsa.table {
    %true = arith.constant true
    %builder = dsa.create_ds("value:") -> !dsa.table_builder<tuple<tuple<i1, i32>>>
    %nullable = util.pack %is_null, %payload : i1, i32 -> tuple<i1, i32>
    dsa.ds_append %builder : !dsa.table_builder<tuple<tuple<i1, i32>>>, %nullable : tuple<i1, i32>, %true
    dsa.next_row %builder : <tuple<tuple<i1, i32>>>
    %table = dsa.finalize %builder : !dsa.table_builder<tuple<tuple<i1, i32>>> -> !dsa.table
    return %table : !dsa.table
  }
}
)mlir");
    REQUIRE(nullableTableBuilderAppend);
    REQUIRE(runDSAToStd(f.ctx, *nullableTableBuilderAppend));
    const std::string nullableAppendLowered = moduleToString(*nullableTableBuilderAppend);
    requireNotContains(nullableAppendLowered, "dsa.ds_append");
    requireContains(nullableAppendLowered, "TableBuilder8addInt32");

    auto nullableTableBuilderAppendAfterPlainColumn = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_table_builder_append_after_plain_column(%id: i32, %is_null: i1, %payload: i32) -> !dsa.table {
    %true = arith.constant true
    %builder = dsa.create_ds("id:;value:") -> !dsa.table_builder<tuple<i32, tuple<i1, i32>>>
    %nullable = util.pack %is_null, %payload : i1, i32 -> tuple<i1, i32>
    dsa.ds_append %builder : !dsa.table_builder<tuple<i32, tuple<i1, i32>>>, %id : i32, %true
    dsa.ds_append %builder : !dsa.table_builder<tuple<i32, tuple<i1, i32>>>, %nullable : tuple<i1, i32>, %true
    dsa.next_row %builder : <tuple<i32, tuple<i1, i32>>>
    %table = dsa.finalize %builder : !dsa.table_builder<tuple<i32, tuple<i1, i32>>> -> !dsa.table
    return %table : !dsa.table
  }
}
)mlir");
    REQUIRE(nullableTableBuilderAppendAfterPlainColumn);
    REQUIRE(runDSAToStd(f.ctx, *nullableTableBuilderAppendAfterPlainColumn));
    const std::string twoColumnAppendLowered = moduleToString(*nullableTableBuilderAppendAfterPlainColumn);
    requireNotContains(twoColumnAppendLowered, "dsa.ds_append");
    REQUIRE(countSubstring(twoColumnAppendLowered, "TableBuilder8addInt32") >= 3);

    auto nonNullableNull = parseModule(f.ctx, R"mlir(
module {
  func.func @bad() {
    %null_i = db.null : !db.pg_int4
    return
  }
}
)mlir",
                                       false);
    REQUIRE(!nonNullableNull);

    auto nullableConstant = parseModule(f.ctx, R"mlir(
module {
  func.func @bad() {
    %i = db.constant(42 : i32) : !db.pg_int4<nullable>
    return
  }
}
)mlir",
                                        false);
    REQUIRE(!nullableConstant);

    PG_RETURN_VOID();
}
