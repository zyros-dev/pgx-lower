// to avoid PostgreSQL's 'restrict' macro pollution

// Basic MLIR infrastructure first
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Heavy template instantiation isolated here
#include <mlir/InitAllPasses.h>

// Dialect registration includes
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Error handling includes
#include "execution/error_handling.h"
#include "execution/logging.h"

// RelAlg includes last to avoid interface conflicts  
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

// Pass registration includes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Transforms/CustomPasses.h"
#include "mlir/Dialect/DB/Passes.h"

// NOW include headers that may bring in PostgreSQL (after MLIR is safe)
#include "execution/mlir_runner.h"
#include "execution/logging.h"

// Pass registration function - isolated to minimize template explosion impact
extern "C" void initialize_mlir_passes() {
    try {
       mlir::registerAllPasses();
       mlir::relalg::registerRelAlgConversionPasses();
       mlir::relalg::registerQueryOptimizationPasses();
       mlir::db::registerDBConversionPasses();
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::dsa::createLowerToStdPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::relalg::createDetachMetaDataPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSinkOpPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSimplifyMemrefsPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSimplifyArithmeticsPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::db::createSimplifyToArithPass();
       });
    } catch (const std::exception& e) {
        PGX_ERROR("Pass registration failed: " + std::string(e.what()));
    } catch (...) {
        PGX_ERROR("Pass registration failed with unknown exception");
    }
}

namespace mlir_runner {

// Extended MLIR context setup for pipeline execution - loads all required dialects
bool setupMLIRContextForJIT(::mlir::MLIRContext& context) {
    if (!initialize_mlir_context(context)) {
        PGX_ERROR("Failed to initialize MLIR context and dialects");
        return false;
    }
    
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
        PGX_ERROR("Failed to initialize MLIR context: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error during MLIR context initialization");
        return false;
    }
}

} // namespace mlir_runner