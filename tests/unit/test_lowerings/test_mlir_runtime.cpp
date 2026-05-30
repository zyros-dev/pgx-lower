#include <gtest/gtest.h>

#include "pgx-lower/execution/mlir_runtime.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Target/TargetMachine.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

using pgx_lower::execution::get_mlir_runtime;

TEST(MLIRRuntimeSingleton, SameInstanceAcrossCalls) {
    auto& rt1 = get_mlir_runtime();
    auto& rt2 = get_mlir_runtime();
    EXPECT_EQ(&rt1, &rt2);
    EXPECT_EQ(&rt1.context, &rt2.context);
}

TEST(MLIRRuntimeSingleton, DialectsPreloaded) {
    auto& rt = get_mlir_runtime();
    EXPECT_NE(rt.context.getLoadedDialect<mlir::db::DBDialect>(), nullptr);
    EXPECT_NE(rt.context.getLoadedDialect<mlir::dsa::DSADialect>(), nullptr);
    EXPECT_NE(rt.context.getLoadedDialect<mlir::util::UtilDialect>(), nullptr);
    EXPECT_NE(rt.context.getLoadedDialect<mlir::relalg::RelAlgDialect>(), nullptr);
}

TEST(MLIRRuntimeSingleton, TargetMachineAvailable) {
    auto& rt = get_mlir_runtime();
    EXPECT_NE(rt.target_machine.get(), nullptr);
}
