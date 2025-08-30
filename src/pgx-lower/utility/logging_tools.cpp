#include "logging_tools.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace pgx_lower {
namespace utility {

void logMLIRModuleVerbose(::mlir::ModuleOp module, const std::string& context) {
}

void logMLIRModuleFullyExpanded(::mlir::ModuleOp module, const std::string& context) {
}

void logMLIRModuleCompact(::mlir::ModuleOp module, const std::string& context) {
    std::string moduleStr;
    llvm::raw_string_ostream stream(moduleStr);
    module.print(stream);
    stream.flush();
    
    std::istringstream moduleStream(moduleStr);
    std::string line;
    while (std::getline(moduleStream, line)) {
        PGX_LOG(GENERAL, DEBUG, "[MLIR] %s", line.c_str());
    }
}

void logMLIROperation(::mlir::Operation* op, const std::string& context) {
    std::string opStr;
    llvm::raw_string_ostream opStream(opStr);
    op->print(opStream);
    opStream.flush();
    PGX_LOG(GENERAL, DEBUG, "[MLIR] %s: %s", context.c_str(), opStr.c_str());
}

void logMLIROperationHierarchy(::mlir::Operation* op, const std::string& context, int depth) {
}

} // namespace utility
} // namespace pgx_lower