extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
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

struct Fixture {
    mlir::MLIRContext ctx;

    Fixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto parseModule(mlir::MLIRContext& ctx, llvm::StringRef moduleText) -> mlir::OwningOpRef<mlir::ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(moduleText), llvm::SMLoc());
    return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
}

auto moduleToString(mlir::ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    return output;
}

auto runDBToStd(mlir::MLIRContext& ctx, mlir::ModuleOp module) -> bool {
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::db::createLowerToStdPass());
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

} // namespace

PGX_TEST_FN(pg_bool_predicate_truth_lowers_non_null_pg_bool_to_i1) {
    Fixture f;
    auto module = parseModule(f.ctx, R"mlir(
module {
  func.func @truth(%value: !db.pg_bool) -> i1 {
    %truth = db.derive_truth %value : !db.pg_bool
    return %truth : i1
  }
}
)mlir");
    REQUIRE(module);
    REQUIRE(runDBToStd(f.ctx, *module));

    const auto printed = moduleToString(*module);
    requireContains(printed, "func.func @truth(%arg0: i1) -> i1");
    requireNotContains(printed, "db.derive_truth");
    requireNotContains(printed, "!db.pg_bool");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_bool_predicate_truth_lowers_nullable_pg_bool_null_to_false) {
    Fixture f;
    auto module = parseModule(f.ctx, R"mlir(
module {
  func.func @nullable_truth(%value: !db.pg_bool<nullable>) -> i1 {
    %truth = db.derive_truth %value : !db.pg_bool<nullable>
    return %truth : i1
  }
}
)mlir");
    REQUIRE(module);
    REQUIRE(runDBToStd(f.ctx, *module));

    const auto printed = moduleToString(*module);
    requireContains(printed, "tuple<i1, i1>");
    requireContains(printed, "util.unpack");
    requireContains(printed, "arith.andi");
    requireNotContains(printed, "db.derive_truth");
    requireNotContains(printed, "!db.pg_bool");
    PG_RETURN_VOID();
}
