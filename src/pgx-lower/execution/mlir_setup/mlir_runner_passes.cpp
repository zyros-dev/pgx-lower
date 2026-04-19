#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

#include <mlir/InitAllPasses.h>

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "lingodb/mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/mlir/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/mlir/Dialect/RelAlg/Passes.h"
#include "lingodb/mlir/Transforms/CustomPasses.h"
#include "lingodb/mlir/Dialect/DB/Passes.h"

#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/utility/logging.h"

extern "C" void initialize_mlir_passes() {
    try {
        mlir::registerAllPasses();
        mlir::relalg::registerRelAlgConversionPasses();
        mlir::relalg::registerQueryOptimizationPasses();
        mlir::db::registerDBConversionPasses();
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::dsa::createLowerToStdPass(); });
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::relalg::createDetachMetaDataPass(); });
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::createSinkOpPass(); });
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::createSimplifyMemrefsPass(); });
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::createSimplifyArithmeticsPass(); });
        ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> { return mlir::db::createSimplifyToArithPass(); });
    } catch (const std::exception& e) {
        PGX_ERROR("Pass registration failed: %s", e.what());
    } catch (...) {
        PGX_ERROR("Pass registration failed with unknown exception");
    }
}

// setupMLIRContextForJIT + initialize_mlir_context used to live here.
// Spec 01 hoisted all per-query MLIR-context setup into the
// MLIRRuntime singleton (mlir_runtime.cpp). The unit-test harness
// StandalonePipelineTester loads dialects inline for its own fresh
// context.