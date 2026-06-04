extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define ASSERT_CONTAINS(haystack, needle) \
    do { if ((haystack).find(needle) == std::string::npos) \
        elog(ERROR, "%s:%d expected to find '%s'", __FILE__, __LINE__, needle); } while (0)
#define ASSERT_NOT_CONTAINS(haystack, needle) \
    do { if ((haystack).find(needle) != std::string::npos) \
        elog(ERROR, "%s:%d unexpected '%s' present", __FILE__, __LINE__, needle); } while (0)

namespace {

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    Fixture() : builder(&ctx) {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToStart(module.getBody());
    }

    bool runDBToStd() {
        mlir::PassManager pm(&ctx);
        pm.addPass(mlir::db::createLowerToStdPass());
        return mlir::succeeded(pm.run(module));
    }

    std::string asString() {
        std::string s{};
        llvm::raw_string_ostream os(s);
        module.print(os);
        return s;
    }
};

}  // namespace

PGX_TEST_FN(boolean_not_lowering) {
    Fixture f;
    auto fn = f.builder.create<mlir::func::FuncOp>(
        f.builder.getUnknownLoc(), "test_not",
        f.builder.getFunctionType({f.builder.getI1Type()}, {f.builder.getI1Type()}));
    auto* block = fn.addEntryBlock();
    f.builder.setInsertionPointToStart(block);
    auto notOp = f.builder.create<mlir::db::NotOp>(f.builder.getUnknownLoc(), block->getArgument(0));
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc(), notOp.getResult());
    if (!f.runDBToStd()) elog(ERROR, "DBToStd pass failed");
    std::string ir = f.asString();
    ASSERT_CONTAINS(ir, "arith.cmpi eq");
    ASSERT_NOT_CONTAINS(ir, "db.not");
    PG_RETURN_VOID();
}

PGX_TEST_FN(boolean_and_lowering) {
    Fixture f;
    auto fn = f.builder.create<mlir::func::FuncOp>(
        f.builder.getUnknownLoc(), "test_and",
        f.builder.getFunctionType({f.builder.getI1Type(), f.builder.getI1Type()}, {f.builder.getI1Type()}));
    auto* block = fn.addEntryBlock();
    f.builder.setInsertionPointToStart(block);
    auto andOp = f.builder.create<mlir::db::AndOp>(
        f.builder.getUnknownLoc(), f.builder.getI1Type(),
        mlir::ValueRange{block->getArgument(0), block->getArgument(1)});
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc(), andOp.getResult());
    if (!f.runDBToStd()) elog(ERROR, "DBToStd pass failed");
    std::string ir = f.asString();
    ASSERT_CONTAINS(ir, "arith.andi");
    ASSERT_NOT_CONTAINS(ir, "db.and");
    ASSERT_NOT_CONTAINS(ir, "arith.select");
    PG_RETURN_VOID();
}

PGX_TEST_FN(boolean_or_lowering) {
    Fixture f;
    auto fn = f.builder.create<mlir::func::FuncOp>(
        f.builder.getUnknownLoc(), "test_or",
        f.builder.getFunctionType({f.builder.getI1Type(), f.builder.getI1Type()}, {f.builder.getI1Type()}));
    auto* block = fn.addEntryBlock();
    f.builder.setInsertionPointToStart(block);
    auto orOp = f.builder.create<mlir::db::OrOp>(
        f.builder.getUnknownLoc(), f.builder.getI1Type(),
        mlir::ValueRange{block->getArgument(0), block->getArgument(1)});
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc(), orOp.getResult());
    if (!f.runDBToStd()) elog(ERROR, "DBToStd pass failed");
    std::string ir = f.asString();
    ASSERT_CONTAINS(ir, "arith.ori");
    ASSERT_NOT_CONTAINS(ir, "db.or");
    ASSERT_NOT_CONTAINS(ir, "arith.select");
    PG_RETURN_VOID();
}

PGX_TEST_FN(boolean_complex_expression) {
    Fixture f;
    auto fn = f.builder.create<mlir::func::FuncOp>(
        f.builder.getUnknownLoc(), "test_complex",
        f.builder.getFunctionType({f.builder.getI1Type(), f.builder.getI1Type(), f.builder.getI1Type()},
                                  {f.builder.getI1Type()}));
    auto* block = fn.addEntryBlock();
    f.builder.setInsertionPointToStart(block);
    auto a = block->getArgument(0);
    auto b = block->getArgument(1);
    auto c = block->getArgument(2);
    auto andOp = f.builder.create<mlir::db::AndOp>(
        f.builder.getUnknownLoc(), f.builder.getI1Type(), mlir::ValueRange{a, b});
    auto notOp = f.builder.create<mlir::db::NotOp>(f.builder.getUnknownLoc(), c);
    auto orOp = f.builder.create<mlir::db::OrOp>(
        f.builder.getUnknownLoc(), f.builder.getI1Type(),
        mlir::ValueRange{andOp.getResult(), notOp.getResult()});
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc(), orOp.getResult());
    if (!f.runDBToStd()) elog(ERROR, "DBToStd pass failed");
    std::string ir = f.asString();
    ASSERT_CONTAINS(ir, "arith.andi");
    ASSERT_CONTAINS(ir, "arith.cmpi eq");
    ASSERT_CONTAINS(ir, "arith.ori");
    ASSERT_NOT_CONTAINS(ir, "db.and");
    ASSERT_NOT_CONTAINS(ir, "db.or");
    ASSERT_NOT_CONTAINS(ir, "db.not");
    ASSERT_NOT_CONTAINS(ir, "arith.select");
    ASSERT_NOT_CONTAINS(ir, "util.unpack");
    ASSERT_NOT_CONTAINS(ir, "util.pack");
    PG_RETURN_VOID();
}
