#include "pgx-lower/execution/jit_execution_engine.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <filesystem>
#include <array>
#include <cstdlib>
#include <chrono>

#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "miscadmin.h"
}
#endif

namespace llvm {
class Module;
}
namespace mlir_runner {
extern void dumpLLVMIR(llvm::Module* module, const std::string& title, pgx_lower::log::Category phase);
}

namespace pgx_lower::execution {

JITEngine::JITEngine(llvm::CodeGenOptLevel opt_level)
: opt_level_(opt_level) {
    setup_llvm_target();
}

bool JITEngine::compile(mlir::ModuleOp module) {
    if (!module) {
        PGX_ERROR("Null module provided");
        return false;
    }

#ifndef PGX_RELEASE_MODE
    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Module verification failed");
        return false;
    }
#endif

    PGX_LOG(JIT, IO, "JIT Compile IN: MLIR Module (opt_level=%d)", static_cast<int>(opt_level_));

    register_dialects(module);

    const auto start_time = std::chrono::high_resolution_clock::now();

    // Store lambdas to avoid dangling pointers
    auto module_builder = create_mlir_to_llvm_translator();
    auto transformer = create_llvm_optimizer();

    mlir::ExecutionEngineOptions options;
    options.llvmModuleBuilder = module_builder;
    options.transformer = transformer;
    options.jitCodeGenOptLevel = opt_level_;
    options.enableObjectDump = true;

    auto maybe_engine = mlir::ExecutionEngine::create(module, options);
    if (!maybe_engine) {
        PGX_ERROR("Failed to create ExecutionEngine");
        return false;
    }

    engine_ = std::move(*maybe_engine);

    if (!lookup_functions()) {
        PGX_WARNING("Initial function lookup failed, trying static linking");
        if (!link_static()) {
            PGX_ERROR("Function lookup and static linking both failed");
            return false;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    compiled_ = true;
    PGX_LOG(JIT, IO, "JIT Compile OUT: Native code ready (compilation took %.2f ms)", duration / 1000.0);
    return true;
}

bool JITEngine::execute(void* estate, void* dest) const {
    if (!compiled_ || !main_fn_) {
        PGX_ERROR("JIT engine not compiled or main function not found");
        return false;
    }
    if (!estate || !dest) {
        PGX_ERROR("Null estate or dest receiver");
        return false;
    }

    PGX_LOG(JIT, IO, "JIT Execute IN: CompiledQuery (estate=%p, dest=%p)", estate, dest);

    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto saved_context = CurrentMemoryContext;

    PG_TRY();
    {
        if (set_context_fn_) {
            PGX_LOG(JIT, DEBUG, "estate pointer value: %p", estate);
            const auto set_ctx = reinterpret_cast<void (*)(void*)>(set_context_fn_);
            set_ctx(estate);
            PGX_LOG(JIT, DEBUG, "Execution context set successfully");
        }

        const auto fn = reinterpret_cast<void (*)()>(main_fn_);
        PGX_LOG(JIT, DEBUG, "About to execute JIT-compiled function at address %p", main_fn_);

        if (g_tuple_streamer.slot) {
            PGX_LOG(JIT, DEBUG, "Before JIT execution: slot=%p, tts_nvalid=%d, tts_tupleDescriptor=%p",
                    g_tuple_streamer.slot, g_tuple_streamer.slot->tts_nvalid, g_tuple_streamer.slot->tts_tupleDescriptor);
        }

        fn();
        PGX_LOG(JIT, DEBUG, "JIT function call returned successfully");

        if (g_tuple_streamer.slot) {
            PGX_LOG(JIT, DEBUG, "After JIT execution: slot=%p, tts_nvalid=%d, tts_tupleDescriptor=%p",
                    g_tuple_streamer.slot, g_tuple_streamer.slot->tts_nvalid, g_tuple_streamer.slot->tts_tupleDescriptor);
        }

        CHECK_FOR_INTERRUPTS();
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(saved_context);

        auto* edata = CopyErrorData();
        PGX_ERROR("PostgreSQL exception during JIT execution: %s (SQLSTATE: %s, detail: %s, hint: %s)",
                  edata->message ? edata->message : "<no message>",
                  edata->sqlerrcode ? unpack_sql_state(edata->sqlerrcode) : "<no code>",
                  edata->detail ? edata->detail : "<no detail>", edata->hint ? edata->hint : "<no hint>");
        FreeErrorData(edata);

        PG_RE_THROW();
    }
    PG_END_TRY();

    MemoryContextSwitchTo(saved_context);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    PGX_LOG(JIT, DEBUG, "JIT execution took %.3f ms", duration.count() / 1000.0);
    PGX_LOG(JIT, IO, "JIT Execute OUT: Query completed successfully (%.2f ms)", duration.count() / 1000.0);

    return true;
}

void JITEngine::setup_llvm_target() {
    static bool initialized = false;
    if (!initialized) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        initialized = true;

        const auto target_triple = llvm::sys::getDefaultTargetTriple();
        PGX_LOG(JIT, DEBUG, "LLVM target triple: %s", target_triple.c_str());
    }
}

void JITEngine::register_dialects(mlir::ModuleOp module) {
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);

    module->getContext()->appendDialectRegistry(registry);
    mlir::registerLLVMDialectTranslation(*module->getContext());
}

std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>
JITEngine::create_mlir_to_llvm_translator() {
    return [](mlir::Operation* op, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
        if (!op) {
            PGX_ERROR("Null operation in module translation");
            return nullptr;
        }

        auto module = mlir::cast<mlir::ModuleOp>(op);

        size_t input_ops_count = 0;
        module.walk([&](mlir::Operation* operation) { input_ops_count++; });

        PGX_LOG(JIT, IO, "MLIR→LLVM IN: Standard MLIR Module with %zu operations", input_ops_count);

        mlir::registerLLVMDialectTranslation(*module->getContext());
        auto llvm_module = mlir::translateModuleToLLVMIR(module, context, "PostgreSQLJITModule");
        if (!llvm_module) {
            PGX_ERROR("Failed to translate MLIR to LLVM IR");
            return nullptr;
        }

#ifndef PGX_RELEASE_MODE
        std::string verify_error;
        llvm::raw_string_ostream verify_stream(verify_error);
        if (llvm::verifyModule(*llvm_module, &verify_stream)) {
            verify_stream.flush();
            PGX_ERROR("LLVM module verification failed: %s", verify_error.c_str());
            return nullptr;
        }
#endif

        for (auto& func : llvm_module->functions()) {
            std::string func_name = func.getName().str();
            if (func_name == "main" || func_name == "_mlir_ciface_main") {
                func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                func.setVisibility(llvm::GlobalValue::DefaultVisibility);
                func.setDSOLocal(true);
            }
        }

        const size_t output_func_count = llvm_module->size();
        size_t output_inst_count = 0;
        for (const auto& func : *llvm_module) {
            for (const auto& bb : func) {
                output_inst_count += bb.size();
            }
        }

        PGX_LOG(JIT, IO, "MLIR→LLVM OUT: LLVM IR Module with %zu functions, %zu instructions", output_func_count,
                output_inst_count);

        return llvm_module;
    };
}

std::function<llvm::Error(llvm::Module*)> JITEngine::create_llvm_optimizer() const {
    auto opt_level = opt_level_;

    return [opt_level](llvm::Module* module) -> llvm::Error {
        PGX_LOG(JIT, DEBUG, "Running optimization lambda (opt_level=%d)", static_cast<int>(opt_level));

        if (opt_level == llvm::CodeGenOptLevel::None) {
            PGX_LOG(JIT, DEBUG, "Optimization disabled");
            return llvm::Error::success();
        }

        try {
            // Install LLVM fatal error handler
            static bool handler_installed = false;
            if (!handler_installed) {
                llvm::install_fatal_error_handler([](void* user_data, const char* reason, bool gen_crash_diag) {
                    PGX_ERROR("LLVM FATAL ERROR: %s (gen_crash_diag=%d)", reason, gen_crash_diag);
                });
                handler_installed = true;
            }

            llvm::TargetMachine* TM = nullptr;
            std::string triple = module->getTargetTriple();
            if (triple.empty()) {
                triple = llvm::sys::getDefaultTargetTriple();
                module->setTargetTriple(triple);
            }

            std::string error;
            const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
            if (target) {
                llvm::TargetOptions target_options;
                TM = target->createTargetMachine(triple, llvm::sys::getHostCPUName(),
                                                 "", // Features
                                                 target_options, llvm::Reloc::PIC_);
                PGX_LOG(JIT, DEBUG, "Created TargetMachine for triple: %s", triple.c_str());
            } else {
                PGX_LOG(JIT, DEBUG, "Failed to create TargetMachine: %s", error.c_str());
            }

            llvm::PipelineTuningOptions PTO;
            PTO.LoopUnrolling = false;
            PTO.LoopVectorization = false;
            PTO.SLPVectorization = false;

            llvm::PassBuilder PB(TM, PTO);
            llvm::LoopAnalysisManager LAM;
            llvm::FunctionAnalysisManager FAM;
            llvm::CGSCCAnalysisManager CGAM;
            llvm::ModuleAnalysisManager MAM;

            PGX_LOG(JIT, DEBUG, "Registering all default analyses");
            PB.registerModuleAnalyses(MAM);
            PB.registerCGSCCAnalyses(CGAM);
            PB.registerFunctionAnalyses(FAM);
            PB.registerLoopAnalyses(LAM);
            PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

            PGX_LOG(JIT, DEBUG, "Building function pass pipeline");
            llvm::FunctionPassManager FPM;

            FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
            FPM.addPass(llvm::InstCombinePass());

            FPM.addPass(llvm::PromotePass());
            FPM.addPass(llvm::InstCombinePass());

            FPM.addPass(llvm::createFunctionToLoopPassAdaptor(llvm::LICMPass(llvm::LICMOptions()),
                                                              /*UseMemorySSA=*/true));

            FPM.addPass(llvm::ReassociatePass());
            FPM.addPass(llvm::GVNPass());
            FPM.addPass(llvm::SimplifyCFGPass());

            PGX_LOG(JIT, DEBUG, "Running passes on %zu functions", module->size());
            for (auto& func : *module) {
                if (func.isDeclaration()) {
                    PGX_LOG(JIT, DEBUG, "Skipping external function: %s", func.getName().str().c_str());
                    continue;
                }
                if (!func.hasOptNone()) {
                    PGX_LOG(JIT, DEBUG, "Running passes on function: %s", func.getName().str().c_str());
                    FPM.run(func, FAM);
                }
            }

            PGX_LOG(JIT, DEBUG, "Optimization passes completed successfully");
            mlir_runner::dumpLLVMIR(module, "LLVM IR AFTER OPTIMIZATION PASSES", log::Category::JIT);

            return llvm::Error::success();
        } catch (const std::exception& e) {
            PGX_ERROR("Exception in optimization lambda: %s", e.what());
            return llvm::make_error<llvm::StringError>("Optimization failed: " + std::string(e.what()),
                                                       llvm::inconvertibleErrorCode());
        } catch (...) {
            PGX_ERROR("Unknown exception in optimization lambda");
            return llvm::make_error<llvm::StringError>("Optimization failed with unknown exception",
                                                       llvm::inconvertibleErrorCode());
        }
    };
}

bool JITEngine::lookup_functions() {
    auto main_lookup = engine_->lookup("main");
    if (!main_lookup) {
        main_lookup = engine_->lookup("_mlir_ciface_main");
    }

    if (main_lookup) {
        main_fn_ = main_lookup.get();
        PGX_LOG(JIT, DEBUG, "Found main function at: %p", main_fn_);
    } else {
        PGX_WARNING("Main function not found via ExecutionEngine::lookup");
        return false;
    }

    auto ctx_lookup = engine_->lookup("rt_set_execution_context");
    if (ctx_lookup) {
        set_context_fn_ = ctx_lookup.get();
        PGX_LOG(JIT, DEBUG, "Found rt_set_execution_context at: %p", set_context_fn_);
    } else {
        PGX_WARNING("rt_set_execution_context function not found");
        return false;
    }

    return main_fn_ != nullptr && set_context_fn_ != nullptr;
}

bool JITEngine::link_static() {
    const std::string obj_path = "/tmp/pgx_jit_module.o";
    const std::string so_path = "/tmp/pgx_jit_module.so";

    PGX_LOG(JIT, DEBUG, "Attempting static linking");

    if (!dump_object_file(obj_path)) {
        return false;
    }

    if (!compile_to_shared_library(obj_path, so_path)) {
        return false;
    }

    void* handle = load_shared_library(so_path);
    if (!handle) {
        return false;
    }

    return lookup_symbols_from_library(handle);
}

bool JITEngine::dump_object_file(const std::string& path) const {
    try {
        engine_->dumpToObjectFile(path);
        PGX_LOG(JIT, DEBUG, "Dumped object file to: %s", path.c_str());
        return true;
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to dump object file: %s", e.what());
        return false;
    }
}

bool JITEngine::compile_to_shared_library(const std::string& obj_path, const std::string& so_path) {
    const std::string cmd = "g++ -shared -fPIC -Wl,--unresolved-symbols=ignore-all -o " + so_path + " " + obj_path
                            + " 2>&1";

    auto* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) {
        PGX_ERROR("Failed to compile JIT object to shared library");
        return false;
    }

    std::array<char, 256> buffer;
    std::string result;
    while (!std::feof(pipe)) {
        const auto bytes = std::fread(buffer.data(), 1, buffer.size(), pipe);
        result.append(buffer.data(), bytes);
    }

    const auto rc = ::pclose(pipe);
    if (WEXITSTATUS(rc)) {
        PGX_ERROR("Compilation failed: %s", result.c_str());
        return false;
    }

    PGX_LOG(JIT, DEBUG, "Compiled shared library: %s", so_path.c_str());
    return true;
}

void* JITEngine::load_shared_library(const std::string& path) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);

    if (!handle) {
        const char* dl_error = dlerror();
        PGX_ERROR("Cannot load shared library: %s", dl_error ? dl_error : "unknown error");
        return nullptr;
    }

    PGX_LOG(JIT, DEBUG, "Loaded shared library: %s", path.c_str());
    return handle;
}

bool JITEngine::lookup_symbols_from_library(void* handle) {
    main_fn_ = dlsym(handle, "main");
    if (!main_fn_) {
        main_fn_ = dlsym(handle, "_mlir_ciface_main");
    }

    if (!main_fn_) {
        const char* dl_error = dlerror();
        PGX_ERROR("Cannot find main function via dlsym: %s", dl_error ? dl_error : "unknown");
        return false;
    }
    PGX_LOG(JIT, DEBUG, "Found main function via dlsym at: %p", main_fn_);

    set_context_fn_ = dlsym(handle, "rt_set_execution_context");
    if (!set_context_fn_) {
        const char* dl_error = dlerror();
        PGX_WARNING("Cannot find rt_set_execution_context via dlsym: %s", dl_error ? dl_error : "unknown");
        return false;
    }
    PGX_LOG(JIT, DEBUG, "Found rt_set_execution_context via dlsym at: %p", set_context_fn_);

    return main_fn_ != nullptr && set_context_fn_ != nullptr;
}

} // namespace pgx_lower::execution
