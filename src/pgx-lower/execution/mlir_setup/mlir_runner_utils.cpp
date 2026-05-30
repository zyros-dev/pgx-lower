#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"
#include <sstream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <mlir/Pass/Pass.h>

namespace mlir_runner {

void dump_module_with_stats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase) {
#ifndef PGX_RELEASE_MODE
    if (!module) {
        PGX_WARNING("dumpModuleWithStats: Module is null for title: %s", title.c_str());
        return;
    }

    auto phase_log = [&](const char* fmt, auto... args) {
        ::pgx_lower::log::log(phase, ::pgx_lower::log::Level::IR, __FILE__, __LINE__, fmt, args...);
    };
    auto timestamp = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);

    std::stringstream filename;
    filename << "/tmp/pgx_ir/pgx_lower_" << title << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".mlir";

    try {
        std::map<std::string, int> dialect_counts;
        std::map<std::string, int> operation_counts;
        std::map<std::string, int> type_counts;
        std::map<std::string, int> attribute_counts;
        int total_operations = 0;
        int total_blocks = 0;
        int total_regions = 0;
        int total_values = 0;

        module.walk([&](::mlir::Operation* op) {
            if (!op) {
                return;
}

            total_operations++;

            std::string dialect_name = op->getName().getDialectNamespace().str();
            if (dialect_name.empty()) {
                dialect_name = "builtin";
}
            dialect_counts[dialect_name]++;

            std::string const op_name = op->getName().getStringRef().str();
            operation_counts[op_name]++;

            total_regions += op->getNumRegions();
            for (auto& region : op->getRegions()) {
                total_blocks += region.getBlocks().size();
            }

            total_values += op->getNumResults();

            for (auto result : op->getResults()) {
                std::string type_name = "unknown";
                llvm::raw_string_ostream stream(type_name);
                result.getType().print(stream);
                type_counts[type_name]++;
            }

            for (auto attr : op->getAttrs()) {
                std::string const attr_type = attr.getName().str();
                attribute_counts[attr_type]++;
            }
        });

        phase_log("\n\n======= %s =======", title.c_str());
        std::stringstream time_str;
        time_str << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        phase_log("Timestamp: %s", time_str.str().c_str());
        phase_log("Output file: %s", filename.str().c_str());

        phase_log("Module Statistics:");
        phase_log("  Total Operations: %d", total_operations);
        phase_log("  Total Blocks: %d", total_blocks);
        phase_log("  Total Regions: %d", total_regions);
        phase_log("  Total Values: %d", total_values);

        try {
            std::string module_str;
            llvm::raw_string_ostream stream(module_str);
            module.print(stream);

            std::stringstream formatted_mlir;
            formatted_mlir << "\n=== MLIR MODULE CONTENT: " << title << " ===\n";

            std::stringstream ss(module_str);
            std::string line;
            int line_num = 1;
            while (std::getline(ss, line)) {
                formatted_mlir << std::setw(3) << line_num << ": " << line << "\n";
                line_num++;
            }
            formatted_mlir << "=== END MLIR MODULE CONTENT ===\n";

            phase_log("%s", formatted_mlir.str().c_str());

        } catch (const std::exception& e) {
            PGX_ERROR("Failed to print MLIR module: %s", e.what());
        }

        bool const is_valid = ::mlir::succeeded(::mlir::verify(module));
        std::ofstream file(filename.str());
        if (file.is_open()) {
            file << "// MLIR Module Debug Dump: " << title << "\n";
            std::stringstream gen_time;
            gen_time << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            file << "// Generated: " << gen_time.str() << "\n";
            file << "// Total Operations: " << total_operations << "\n";
            file << "// Module Valid: " << (is_valid ? "YES" : "NO") << "\n\n";

            std::string module_str;
            llvm::raw_string_ostream stream(module_str);
            module.print(stream);
            file << module_str;
            file.close();

            phase_log("Module dumped to: %s", filename.str().c_str());
        }
        else {
            PGX_WARNING("Failed to open file for writing: %s", filename.str().c_str());
        }

        phase_log("=== End Module Debug Dump ===");
        phase_log("\n\n");

    } catch (const std::exception& e) {
        PGX_ERROR("Exception in dumpModuleWithStats: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception in dumpModuleWithStats");
    }
#endif // PGX_RELEASE_MODE
}

void dump_llvmir(llvm::Module* module, const std::string& title, pgx_lower::log::Category phase) {
    if (!module) {
        PGX_WARNING("dumpLLVMIR: Module is null for title: %s", title.c_str());
        return;
    }

    auto phase_log = [&](const char* fmt, auto... args) {
        ::pgx_lower::log::log(phase, ::pgx_lower::log::Level::IR, __FILE__, __LINE__, fmt, args...);
    };

    phase_log("=== %s ===", title.c_str());

    for (auto& func : *module) {
        if (func.getName() == "main") {
            std::string func_str;
            llvm::raw_string_ostream func_stream(func_str);
            func.print(func_stream, nullptr);
            func_stream.flush();
            phase_log("%s", func_str.c_str());
            return;
        }
    }

    PGX_WARNING("dumpLLVMIR: main() function not found in module");
}

class ModuleDumpPass : public mlir::PassWrapper<ModuleDumpPass, mlir::OperationPass<mlir::ModuleOp>> {
private:
    std::string phaseName_;
    ::pgx_lower::log::Category phaseCategory_;

public:
    explicit ModuleDumpPass(std::string  name, ::pgx_lower::log::Category category = ::pgx_lower::log::Category::GENERAL)
        : phaseName_(std::move(name)), phaseCategory_(category) {}

    void runOnOperation() override {
        dump_module_with_stats(getOperation(), phaseName_, phaseCategory_);
    }

    [[nodiscard]] llvm::StringRef getArgument() const override { return "module-dump"; }
    [[nodiscard]] llvm::StringRef getDescription() const override {
        return "Dump MLIR module for debugging";
    }
};

std::unique_ptr<mlir::Pass> create_module_dump_pass(const std::string& phase_name, ::pgx_lower::log::Category category) {
    return std::make_unique<ModuleDumpPass>(phase_name, category);
}

bool validate_module_state(::mlir::ModuleOp module, const std::string& phase) {
#ifndef PGX_RELEASE_MODE
    if (!module || !module.getOperation()) {
        PGX_ERROR("%s: Module operation is null", phase.c_str());
        return false;
    }

    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR("%s: Module verification failed", phase.c_str());
        return false;
    }
#endif
    return true;
}

} // namespace mlir_runner