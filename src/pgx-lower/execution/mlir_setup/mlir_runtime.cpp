#include "pgx-lower/execution/mlir_runtime.h"

#include <memory>
#include <string>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "pgx-lower/utility/logging.h"

namespace pgx_lower::execution {

namespace {

std::unique_ptr<MLIRRuntime> g_runtime;

void init_llvm_targets() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
}

void populate_dialect_registry(mlir::DialectRegistry& registry) {
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
}

void load_dialects(mlir::MLIRContext& context) {
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
    context.getOrLoadDialect<::mlir::db::DBDialect>();
    context.getOrLoadDialect<::mlir::dsa::DSADialect>();
    context.getOrLoadDialect<::mlir::util::UtilDialect>();
}

void install_diagnostic_handler(mlir::MLIRContext& context) {
    context.getDiagEngine().registerHandler([](mlir::Diagnostic& diag) {
        std::string diagStr;
        llvm::raw_string_ostream os(diagStr);

        std::string locStr;
        llvm::raw_string_ostream locOs(locStr);
        diag.getLocation().print(locOs);
        locOs.flush();

        diag.print(os);
        os.flush();

        const char* loc = locStr.empty() ? "unknown" : locStr.c_str();
        switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Error:
                PGX_ERROR("MLIR Error at %s: %s", loc, diagStr.c_str());
                break;
            case mlir::DiagnosticSeverity::Warning:
                PGX_WARNING("MLIR Warning at %s: %s", loc, diagStr.c_str());
                break;
            case mlir::DiagnosticSeverity::Note:
            case mlir::DiagnosticSeverity::Remark:
                PGX_LOG(GENERAL, DEBUG, "MLIR Note at %s: %s", loc, diagStr.c_str());
                break;
        }

        return mlir::success();
    });
}

std::unique_ptr<llvm::TargetMachine> build_target_machine() {
    const std::string triple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return nullptr;
    }

    llvm::TargetOptions target_options;
    llvm::TargetMachine* tm = target->createTargetMachine(
        triple, llvm::sys::getHostCPUName(),
        /*Features=*/"",
        target_options, llvm::Reloc::PIC_);
    return std::unique_ptr<llvm::TargetMachine>(tm);
}

}  // namespace

MLIRRuntime& get_mlir_runtime() {
    if (!g_runtime) {
        init_llvm_targets();

        g_runtime = std::make_unique<MLIRRuntime>();

        populate_dialect_registry(g_runtime->registry);
        g_runtime->context.appendDialectRegistry(g_runtime->registry);
        g_runtime->context.disableMultithreading();
        load_dialects(g_runtime->context);
        mlir::registerLLVMDialectTranslation(g_runtime->context);
        install_diagnostic_handler(g_runtime->context);

        g_runtime->target_machine = build_target_machine();
    }
    return *g_runtime;
}

void shutdown_mlir_runtime() {
    // Must not log via PGX_LOG here — by the time _PG_fini runs, logging
    // GUCs may already be torn down.
    g_runtime.reset();
}

}  // namespace pgx_lower::execution

extern "C" void initialize_mlir_runtime(void) {
    (void)pgx_lower::execution::get_mlir_runtime();
}

extern "C" void shutdown_mlir_runtime_c(void) {
    pgx_lower::execution::shutdown_mlir_runtime();
}
