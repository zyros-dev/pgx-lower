extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
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
