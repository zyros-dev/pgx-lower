#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pgx-lower/utility/logging.h"
#include "llvm/Support/FormatVariadic.h"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pgx_lower {

// Forward declarations
std::unique_ptr<Pass> createConvertToLLVMPass();
std::unique_ptr<Pass> createStandardToLLVMPass();



void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification) {
    PGX_LOG(JIT, DEBUG, "createRelAlgToDBPipeline: Adding relalg to db pipeline");

    if (enableVerification) {
        pm.enableVerifier(true);
    }

    relalg::createLowerRelAlgPipeline(pm);
}

void createDBToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_LOG(JIT, DEBUG, "[createDBToStandardPipeline]: Adding DB to Standard pipeline (Phase 2a)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }

    db::createLowerDBPipeline(pm);
    pm.addPass(createCanonicalizerPass());
}

void createDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_LOG(JIT, DEBUG, "[createDSAToStandardPipeline]: Adding DSA to Standard pipeline (Phase 2b)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }

    auto dsaPass = dsa::createLowerToStdPass();
    if (!dsaPass) {
        PGX_ERROR("createDSAToStandardPipeline: DSA pass creation returned null!");
        return;
    }
    pm.addPass(std::move(dsaPass));
    pm.addPass(createCanonicalizerPass());
}

void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_LOG(JIT, DEBUG, "[createStandardToLLVMPipeline]: Adding DB DSA to Standard pipeline");
    if (enableVerification) {
        pm.enableVerifier(true);
    }

    pm.addPass(std::move(createStandardToLLVMPass()));
    pm.addPass(createCanonicalizerPass());
}

} // namespace pgx_lower
} // namespace mlir
