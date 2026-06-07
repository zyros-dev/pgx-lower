extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "pgx-lower/test/pgx_test_fn.h"

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

auto parseModule(mlir::MLIRContext& ctx, llvm::StringRef moduleText) -> mlir::OwningOpRef<mlir::ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(moduleText), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler handler(sourceMgr, &ctx);
    return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
}

} // namespace

PGX_TEST_FN(pg_legacy_nullable_rejects_pg_wrapper_type) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::db::DBDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();

    auto module = parseModule(ctx, R"mlir(
module {
  func.func @bad(%value: !db.nullable<!db.pg_int4>) {
    return
  }
}
)mlir");

    const bool parserRejectedModule = !module;
    const bool verifierRejectedModule = module && mlir::failed(mlir::verify(*module));
    const bool rejectedPgWrapperType = parserRejectedModule || verifierRejectedModule;
    REQUIRE(rejectedPgWrapperType);

    PG_RETURN_VOID();
}
