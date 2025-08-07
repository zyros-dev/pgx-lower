#include "logging_tools.h"
#include "execution/logging.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace pgx_lower {
namespace utility {

void logMLIRModuleVerbose(mlir::ModuleOp module, const std::string& context) {
    MLIR_PGX_INFO("MLIR", "=== VERBOSE MLIR MODULE DUMP: " + context + " ===");
    
    // Walk all operations and print them individually to get full operation content
    module.walk([&](mlir::Operation* op) {
        std::string opStr;
        llvm::raw_string_ostream opStream(opStr);
        
        // Force full expansion of nested regions - no folding
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        flags.printGenericOpForm(false); // Keep named form but expand regions
        flags.assumeVerified();
        
        op->print(opStream, flags);
        opStream.flush();
        
        // Split operation into lines and log each one
        std::istringstream opStrStream(opStr);
        std::string line;
        int lineNum = 1;
        while (std::getline(opStrStream, line)) {
            std::string logLine = "VERBOSE_OP_LINE_" + std::to_string(lineNum) + ": " + line;
            MLIR_PGX_INFO("MLIR", logLine);
            lineNum++;
        }
        MLIR_PGX_INFO("MLIR", "=== END OPERATION ===");
        return mlir::WalkResult::advance();
    });
    
    MLIR_PGX_INFO("MLIR", "=== END VERBOSE MLIR MODULE DUMP: " + context + " ===");
}

void logMLIRModuleFullyExpanded(mlir::ModuleOp module, const std::string& context) {
    MLIR_PGX_INFO("MLIR", "=== FULLY EXPANDED MLIR MODULE DUMP: " + context + " ===");
    
    std::string moduleStr;
    llvm::raw_string_ostream stream(moduleStr);
    
    // Use printing flags that force full expansion of all regions
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo();
    flags.printGenericOpForm(true); // Use generic form to force region expansion
    flags.assumeVerified();
    flags.elideLargeElementsAttrs(false); // Don't elide large attributes
    flags.skipRegions(false); // Explicitly don't skip regions
    
    module.print(stream, flags);
    stream.flush();
    
    // Split the module string into lines and log each one
    std::istringstream moduleStream(moduleStr);
    std::string line;
    int lineNum = 1;
    while (std::getline(moduleStream, line)) {
        std::string logLine = "EXPANDED_LINE_" + std::to_string(lineNum) + ": " + line;
        MLIR_PGX_INFO("MLIR", logLine);
        lineNum++;
    }
    
    MLIR_PGX_INFO("MLIR", "=== END FULLY EXPANDED MLIR MODULE DUMP: " + context + " ===");
}

void logMLIRModuleCompact(mlir::ModuleOp module, const std::string& context) {
    MLIR_PGX_INFO("MLIR", "=== COMPACT MLIR MODULE DUMP: " + context + " ===");
    
    std::string moduleStr;
    llvm::raw_string_ostream stream(moduleStr);
    module.print(stream);
    stream.flush();
    
    // Split the module string into lines and log each one to avoid PostgreSQL notice limits
    std::istringstream moduleStream(moduleStr);
    std::string line;
    int lineNum = 1;
    while (std::getline(moduleStream, line)) {
        std::string logLine = "MLIR_LINE_" + std::to_string(lineNum) + ": " + line;
        MLIR_PGX_INFO("MLIR", logLine);
        lineNum++;
    }
    
    MLIR_PGX_INFO("MLIR", "=== END COMPACT MLIR MODULE DUMP: " + context + " ===");
}

void logMLIROperation(mlir::Operation* op, const std::string& context) {
    MLIR_PGX_INFO("MLIR", "=== MLIR OPERATION DUMP: " + context + " ===");
    
    std::string opStr;
    llvm::raw_string_ostream opStream(opStr);
    op->print(opStream);
    opStream.flush();
    
    // Split operation into lines and log each one
    std::istringstream opStrStream(opStr);
    std::string line;
    int lineNum = 1;
    while (std::getline(opStrStream, line)) {
        std::string logLine = "OP_LINE_" + std::to_string(lineNum) + ": " + line;
        MLIR_PGX_INFO("MLIR", logLine);
        lineNum++;
    }
    
    MLIR_PGX_INFO("MLIR", "=== END MLIR OPERATION DUMP: " + context + " ===");
}

void logMLIROperationHierarchy(mlir::Operation* op, const std::string& context, int depth) {
    std::string indent(depth * 2, ' ');
    std::string prefix = "HIERARCHY_D" + std::to_string(depth) + ": " + indent;
    
    MLIR_PGX_INFO("MLIR", prefix + "Operation: " + op->getName().getStringRef().str());
    
    // Log operation details
    std::string opStr;
    llvm::raw_string_ostream opStream(opStr);
    op->print(opStream);
    opStream.flush();
    
    // Get just the first line (operation signature)
    std::istringstream opStrStream(opStr);
    std::string firstLine;
    if (std::getline(opStrStream, firstLine)) {
        MLIR_PGX_INFO("MLIR", prefix + firstLine);
    }
    
    // Recursively log nested operations
    for (auto& region : op->getRegions()) {
        for (auto& block : region.getBlocks()) {
            for (auto& nested_op : block.getOperations()) {
                logMLIROperationHierarchy(&nested_op, context, depth + 1);
            }
        }
    }
}

} // namespace utility
} // namespace pgx_lower