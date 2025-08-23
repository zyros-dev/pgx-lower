#include "logging_tools.h"
#include "pgx-lower/execution/logging.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace pgx_lower {
namespace utility {

void logMLIRModuleVerbose(::mlir::ModuleOp module, const std::string& context) {
    // Verbose logging removed - use compact version if needed
}

void logMLIRModuleFullyExpanded(::mlir::ModuleOp module, const std::string& context) {
    // Fully expanded logging removed - use compact version if needed
}

void logMLIRModuleCompact(::mlir::ModuleOp module, const std::string& context) {
    std::string moduleStr;
    llvm::raw_string_ostream stream(moduleStr);
    module.print(stream);
    stream.flush();
    
    std::istringstream moduleStream(moduleStr);
    std::string line;
    while (std::getline(moduleStream, line)) {
        MLIR_PGX_DEBUG("MLIR", line);
    }
}

void logMLIROperation(::mlir::Operation* op, const std::string& context) {
    std::string opStr;
    llvm::raw_string_ostream opStream(opStr);
    op->print(opStream);
    opStream.flush();
    MLIR_PGX_DEBUG("MLIR", context + ": " + opStr);
}

void logMLIROperationHierarchy(::mlir::Operation* op, const std::string& context, int depth) {
    // Hierarchy logging removed - use compact operation logging if needed
}

} // namespace utility
} // namespace pgx_lower