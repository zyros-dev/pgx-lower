#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"
#include <sstream>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <mlir/Pass/Pass.h>

namespace mlir_runner {

void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase) {
    if (!module) {
        PGX_WARNING("dumpModuleWithStats: Module is null for title: %s", title.c_str());
        return;
    }

    auto PHASE_LOG = [&](const char* fmt, auto... args) {
        ::pgx_lower::log::log(phase, ::pgx_lower::log::Level::IR, __FILE__, __LINE__, fmt, args...);
    };
    auto timestamp = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);

    std::stringstream filename;
    filename << "/tmp/pgx_lower_" << title << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".mlir";

    try {
        std::map<std::string, int> dialectCounts;
        std::map<std::string, int> operationCounts;
        std::map<std::string, int> typeCounts;
        std::map<std::string, int> attributeCounts;
        int totalOperations = 0;
        int totalBlocks = 0;
        int totalRegions = 0;
        int totalValues = 0;

        module.walk([&](::mlir::Operation* op) {
            if (!op)
                return;

            totalOperations++;

            std::string dialectName = op->getName().getDialectNamespace().str();
            if (dialectName.empty())
                dialectName = "builtin";
            dialectCounts[dialectName]++;

            std::string opName = op->getName().getStringRef().str();
            operationCounts[opName]++;

            totalRegions += op->getNumRegions();
            for (auto& region : op->getRegions()) {
                totalBlocks += region.getBlocks().size();
            }

            totalValues += op->getNumResults();

            for (auto result : op->getResults()) {
                std::string typeName = "unknown";
                llvm::raw_string_ostream stream(typeName);
                result.getType().print(stream);
                typeCounts[typeName]++;
            }

            for (auto attr : op->getAttrs()) {
                std::string attrType = attr.getName().str();
                attributeCounts[attrType]++;
            }
        });

        PHASE_LOG("\n\n======= %s =======", title.c_str());
        std::stringstream timeStr;
        timeStr << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        PHASE_LOG("Timestamp: %s", timeStr.str().c_str());
        PHASE_LOG("Output file: %s", filename.str().c_str());

        PHASE_LOG("Module Statistics:");
        PHASE_LOG("  Total Operations: %d", totalOperations);
        PHASE_LOG("  Total Blocks: %d", totalBlocks);
        PHASE_LOG("  Total Regions: %d", totalRegions);
        PHASE_LOG("  Total Values: %d", totalValues);

        try {
            std::string moduleStr;
            llvm::raw_string_ostream stream(moduleStr);
            module.print(stream);

            std::stringstream formattedMLIR;
            formattedMLIR << "\n=== MLIR MODULE CONTENT: " << title << " ===\n";

            std::stringstream ss(moduleStr);
            std::string line;
            int lineNum = 1;
            while (std::getline(ss, line)) {
                formattedMLIR << std::setw(3) << lineNum << ": " << line << "\n";
                lineNum++;
            }
            formattedMLIR << "=== END MLIR MODULE CONTENT ===\n";

            PHASE_LOG("%s", formattedMLIR.str().c_str());

        } catch (const std::exception& e) {
            PGX_ERROR("Failed to print MLIR module: %s", e.what());
        }

        bool isValid = ::mlir::succeeded(::mlir::verify(module));
        std::ofstream file(filename.str());
        if (file.is_open()) {
            file << "// MLIR Module Debug Dump: " << title << "\n";
            std::stringstream genTime;
            genTime << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            file << "// Generated: " << genTime.str() << "\n";
            file << "// Total Operations: " << totalOperations << "\n";
            file << "// Module Valid: " << (isValid ? "YES" : "NO") << "\n\n";

            std::string moduleStr;
            llvm::raw_string_ostream stream(moduleStr);
            module.print(stream);
            file << moduleStr;
            file.close();

            PHASE_LOG("Module dumped to: %s", filename.str().c_str());
        }
        else {
            PGX_WARNING("Failed to open file for writing: %s", filename.str().c_str());
        }

        PHASE_LOG("=== End Module Debug Dump ===");
        PHASE_LOG("\n\n");

    } catch (const std::exception& e) {
        PGX_ERROR("Exception in dumpModuleWithStats: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception in dumpModuleWithStats");
    }
}

class ModuleDumpPass : public mlir::PassWrapper<ModuleDumpPass, mlir::OperationPass<mlir::ModuleOp>> {
private:
    std::string phaseName;
    ::pgx_lower::log::Category phaseCategory;

public:
    ModuleDumpPass(const std::string& name, ::pgx_lower::log::Category category = ::pgx_lower::log::Category::GENERAL)
        : phaseName(name), phaseCategory(category) {}

    void runOnOperation() override {
        dumpModuleWithStats(getOperation(), phaseName, phaseCategory);
    }

    llvm::StringRef getArgument() const override { return "module-dump"; }
    llvm::StringRef getDescription() const override {
        return "Dump MLIR module for debugging";
    }
};

std::unique_ptr<mlir::Pass> createModuleDumpPass(const std::string& phaseName, ::pgx_lower::log::Category category) {
    return std::make_unique<ModuleDumpPass>(phaseName, category);
}

bool validateModuleState(::mlir::ModuleOp module, const std::string& phase) {
    if (!module || !module.getOperation()) {
        PGX_ERROR("%s: Module operation is null", phase.c_str());
        return false;
    }

    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR("%s: Module verification failed", phase.c_str());
        return false;
    }

    return true;
}

} // namespace mlir_runner