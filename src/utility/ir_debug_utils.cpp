//===- ir_debug_utils.cpp - MLIR IR debugging utilities -----------------===//

#include "utility/ir_debug_utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <functional>

using namespace mlir;

namespace pgx {
namespace utility {

bool hasCircularIR(ModuleOp module) {
    std::set<uintptr_t> visited;
    std::set<uintptr_t> recursionStack;  // Current path from root to current node
    
    // Manual recursive traversal that properly tracks the path
    std::function<bool(Operation*)> detectCycle = [&](Operation* op) -> bool {
        if (!op) return false;

        auto addr = reinterpret_cast<uintptr_t>(op);

        // Log current recursion stack state
        std::string stackStr = "[";
        for (auto stackAddr : recursionStack) {
            stackStr += std::to_string(stackAddr) + ",";
        }
        if (!recursionStack.empty()) stackStr.pop_back(); // Remove last comma
        stackStr += "]";

        PGX_INFO("Visiting operation: " + op->getName().getStringRef().str() + 
                 " (addr: " + std::to_string(addr) + ") - recursion stack: " + stackStr);

        // CRITICAL: Check if we've seen this op in the CURRENT PATH - cycle detected!
        if (recursionStack.count(addr)) {
            PGX_INFO("Circular IR detected! Operation " + op->getName().getStringRef().str() + 
                     " (addr: " + std::to_string(addr) + ") " +
                     "already in current path - this creates infinite loop");
            return true;
        }

        // If already fully processed (not in current path), skip
        if (visited.count(addr)) {
            PGX_INFO("Already processed: " + op->getName().getStringRef().str() + " (addr: " + std::to_string(addr) + ")");
            return false;
        }

        // Add to current path
        PGX_INFO("Adding to recursion stack: " + op->getName().getStringRef().str() + " (addr: " + std::to_string(addr) + ")");
        recursionStack.insert(addr);

        // Manually traverse all nested operations
        for (auto& region : op->getRegions()) {
            for (auto& block : region.getBlocks()) {
                for (auto& nestedOp : block.getOperations()) {
                    if (detectCycle(&nestedOp)) {
                        return true;  // Cycle found in child
                    }
                }
            }
        }

        // Remove from current path and mark as fully processed
        recursionStack.erase(addr);
        visited.insert(addr);
        
        return false;
    };
    
    // Start traversal from the module
    return detectCycle(module.getOperation());
}

} // namespace utility
} // namespace pgx