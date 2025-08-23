#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"
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

// Module statistics and debugging utilities
void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title) {
    if (!module) {
        PGX_WARNING("dumpModuleWithStats: Module is null for title: " + title);
        return;
    }

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

        // Log comprehensive statistics
        PGX_INFO("\n\n======= " + title + " =======");
        std::stringstream timeStr;
        timeStr << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        PGX_INFO("Timestamp: " + timeStr.str());
        PGX_INFO("Output file: " + filename.str());

        // Overall statistics
        PGX_INFO("Module Statistics:");
        PGX_INFO("  Total Operations: " + std::to_string(totalOperations));
        PGX_INFO("  Total Blocks: " + std::to_string(totalBlocks));
        PGX_INFO("  Total Regions: " + std::to_string(totalRegions));
        PGX_INFO("  Total Values: " + std::to_string(totalValues));

        // Dialect breakdown
        PGX_INFO("Operations by Dialect:");
        for (const auto& [dialect, count] : dialectCounts) {
            PGX_INFO("  " + dialect + ": " + std::to_string(count));
        }

        // Top 10 most frequent operations
        std::vector<std::pair<int, std::string>> opsByFreq;
        for (const auto& [op, count] : operationCounts) {
            opsByFreq.emplace_back(count, op);
        }
        std::sort(opsByFreq.rbegin(), opsByFreq.rend());

        PGX_INFO("Top Operations by Frequency:");
        for (size_t i = 0; i < std::min(size_t(10), opsByFreq.size()); ++i) {
            PGX_INFO("  " + opsByFreq[i].second + ": " + std::to_string(opsByFreq[i].first));
        }

        // Type statistics (top 5)
        std::vector<std::pair<int, std::string>> typesByFreq;
        for (const auto& [type, count] : typeCounts) {
            typesByFreq.emplace_back(count, type);
        }
        std::sort(typesByFreq.rbegin(), typesByFreq.rend());

        PGX_INFO("Top Types by Frequency:");
        for (size_t i = 0; i < std::min(size_t(5), typesByFreq.size()); ++i) {
            PGX_INFO("  " + typesByFreq[i].second + ": " + std::to_string(typesByFreq[i].first));
        }

        bool isValid = ::mlir::succeeded(::mlir::verify(module));
        PGX_INFO("Module Verification: " + std::string(isValid ? "PASSED" : "FAILED"));

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
            PGX_INFO(formattedMLIR.str());

        } catch (const std::exception& e) {
            PGX_ERROR("Failed to print MLIR module: " + std::string(e.what()));
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

            PGX_INFO("Module dumped to: " + filename.str());
        }
        else {
            PGX_WARNING("Failed to open file for writing: " + filename.str());
        }

        PGX_INFO("=== End Module Debug Dump ===");
        PGX_INFO("\n\n");

    } catch (const std::exception& e) {
        PGX_ERROR("Exception in dumpModuleWithStats: " + std::string(e.what()));
    } catch (...) {
        PGX_ERROR("Unknown exception in dumpModuleWithStats");
    }
}

bool validateModuleState(::mlir::ModuleOp module, const std::string& phase) {
    if (!module || !module.getOperation()) {
        PGX_ERROR(phase + ": Module operation is null");
        return false;
    }

    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR(phase + ": Module verification failed");
        return false;
    }

    return true;
}

} // namespace mlir_runner