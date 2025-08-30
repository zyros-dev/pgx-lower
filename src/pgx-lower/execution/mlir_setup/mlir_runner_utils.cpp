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

namespace mlir_runner {

void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase) {
    if (!module) {
        PGX_WARNING("dumpModuleWithStats: Module is null for title: %s", title.c_str());
        return;
    }

    auto PHASE_LOG = [&](const char* fmt, auto... args) {
        ::pgx_lower::log::log(phase, ::pgx_lower::log::Level::DEBUG, fmt, args...);
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

        // Log comprehensive statistics using the passed phase category
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

        // Dialect breakdown
        PHASE_LOG("Operations by Dialect:");
        for (const auto& [dialect, count] : dialectCounts) {
            PHASE_LOG("  %s: %d", dialect.c_str(), count);
        }

        // Top 10 most frequent operations
        std::vector<std::pair<int, std::string>> opsByFreq;
        for (const auto& [op, count] : operationCounts) {
            opsByFreq.emplace_back(count, op);
        }
        std::sort(opsByFreq.rbegin(), opsByFreq.rend());

        PHASE_LOG("Top Operations by Frequency:");
        for (size_t i = 0; i < std::min(size_t(10), opsByFreq.size()); ++i) {
            PHASE_LOG("  %s: %d", opsByFreq[i].second.c_str(), opsByFreq[i].first);
        }

        // Type statistics (top 5)
        std::vector<std::pair<int, std::string>> typesByFreq;
        for (const auto& [type, count] : typeCounts) {
            typesByFreq.emplace_back(count, type);
        }
        std::sort(typesByFreq.rbegin(), typesByFreq.rend());

        PHASE_LOG("Top Types by Frequency:");
        for (size_t i = 0; i < std::min(size_t(5), typesByFreq.size()); ++i) {
            PHASE_LOG("  %s: %d", typesByFreq[i].second.c_str(), typesByFreq[i].first);
        }

        bool isValid = ::mlir::succeeded(::mlir::verify(module));
        PHASE_LOG("Module Verification: %s", isValid ? "PASSED" : "FAILED");

        // Print the actual MLIR code to logs as a formatted block
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

            // Log the entire formatted block
            PHASE_LOG("%s", formattedMLIR.str().c_str());

        } catch (const std::exception& e) {
            PGX_ERROR("Failed to print MLIR module: %s", e.what());
        }

        // Write module to file
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