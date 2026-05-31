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

namespace mlir_runner {

bool setupMLIRContextForJIT(::mlir::MLIRContext& context) {
    if (!initialize_mlir_context(context)) {
        PGX_ERROR("Failed to initialize MLIR context and dialects");
        return false;
    }

    context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
        std::string diagStr;
        llvm::raw_string_ostream os(diagStr);
        
        std::string locStr;
        llvm::raw_string_ostream locOs(locStr);
        diag.getLocation().print(locOs);
        locOs.flush();
        
        diag.print(os);
        os.flush();
        
        switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Error:
            case mlir::DiagnosticSeverity::Warning:
            case mlir::DiagnosticSeverity::Note:
            case mlir::DiagnosticSeverity::Remark:
                PGX_WARNING("MLIR Note at %s: %s",
                       locStr.empty() ? "unknown" : locStr.c_str(), 
                       diagStr.c_str());
                break;
        }
        
        return mlir::success();
    });

    context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
    context.getOrLoadDialect<::mlir::db::DBDialect>();
    context.getOrLoadDialect<::mlir::dsa::DSADialect>();
    context.getOrLoadDialect<::mlir::util::UtilDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    return true;
}

bool initialize_mlir_context(::mlir::MLIRContext& context) {
    try {
        context.disableMultithreading();

        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::mlir::db::DBDialect>();
        context.getOrLoadDialect<::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<::mlir::util::UtilDialect>();

        return true;

    } catch (const std::exception& e) {
        PGX_ERROR("Failed to initialize MLIR context: %s", e.what());
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error during MLIR context initialization");
        return false;
    }
}

} // namespace mlir_runner