#pragma once

#include <memory>
#include <functional>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {
class Module;
class LLVMContext;
class Error;
} // namespace llvm

namespace mlir {
class Operation;
}

namespace pgx_lower::execution {

class JITEngine {
   public:
    explicit JITEngine(llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::Default);
    ~JITEngine() = default;

    JITEngine(const JITEngine&) = delete;
    JITEngine& operator=(const JITEngine&) = delete;
    JITEngine(JITEngine&&) = default;
    JITEngine& operator=(JITEngine&&) = default;

    bool compile(mlir::ModuleOp module);
    bool execute(void* estate, void* dest) const;

   private:
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    void* main_fn_{nullptr};
    void* set_context_fn_{nullptr};
    llvm::CodeGenOptLevel opt_level_;
    bool compiled_{false};

    static void setup_llvm_target();
    static void register_dialects(mlir::ModuleOp module);
    static std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>
    create_mlir_to_llvm_translator();
    std::function<llvm::Error(llvm::Module*)> create_llvm_optimizer() const;

    bool lookup_functions();
    bool link_static();

    bool dump_object_file(const std::string& path) const;
    static bool compile_to_shared_library(const std::string& obj_path, const std::string& so_path);
    static void* load_shared_library(const std::string& path);
    bool lookup_symbols_from_library(void* handle);
};

} // namespace pgx_lower::execution
