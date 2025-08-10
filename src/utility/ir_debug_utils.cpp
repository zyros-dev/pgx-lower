//===- ir_debug_utils.cpp - MLIR IR debugging utilities -----------------===//

#include "utility/ir_debug_utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <iostream>

using namespace mlir;

namespace pgx {
namespace utility {

void logModuleIR(ModuleOp module, const std::string& label) {
    std::cout << "=== " << label << " - Module IR Structure ===" << std::endl;
    std::cout << "WARNING: module.print() may hang if IR is malformed - using basic info only" << std::endl;
    
    // Get basic module info without walking the full structure
    std::cout << "Module address: " << reinterpret_cast<uintptr_t>(module.getOperation()) << std::endl;
    std::cout << "Module has " << module.getBody()->getOperations().size() << " top-level operations" << std::endl;
    
    // Try to print just the operation names without full IR
    std::cout << "Top-level operations:" << std::endl;
    for (auto& op : module.getBody()->getOperations()) {
        std::cout << "  - " << op.getName().getStringRef().str() << std::endl;
    }
    
    std::cout << "=== End " << label << " ===" << std::endl;
}

bool hasCircularIR(ModuleOp module) {
    // Log the module structure before analysis
    logModuleIR(module, "CYCLE_DEBUG");
    
    std::set<uintptr_t> visited;
    std::set<uintptr_t> recursionStack;  // Current path from root to current node
    int operationCount = 0;
    const int MAX_OPERATIONS = 100; // Prevent infinite loops in detector itself

    // Manual recursive traversal that properly tracks the path
    std::function<bool(Operation*)> detectCycle = [&](Operation* op) -> bool {
        if (!op) return false;
        
        // Safety check to prevent infinite loops in detector itself
        if (++operationCount > MAX_OPERATIONS) {
            return false; // Assume no cycles if we hit safety limit
        }

        auto addr = reinterpret_cast<uintptr_t>(op);

        // Log current recursion stack state
        std::string stackStr = "[";
        for (auto stackAddr : recursionStack) {
            stackStr += std::to_string(stackAddr) + ",";
        }
        if (!recursionStack.empty()) stackStr.pop_back(); // Remove last comma
        stackStr += "]";

        std::cout << "[CYCLE_DEBUG] Visiting operation: " << op->getName().getStringRef().str() 
                  << " (addr: " << addr << ") - recursion stack: " << stackStr << std::endl;

        // CRITICAL: Check if we've seen this op in the CURRENT PATH - cycle detected!
        if (recursionStack.count(addr)) {
            std::cout << "[CYCLE_DEBUG] Circular IR detected! Operation " << op->getName().getStringRef().str() 
                      << " (addr: " << addr << ") already in current path - this creates infinite loop" << std::endl;
            return true;
        }

        // If already fully processed (not in current path), skip
        if (visited.count(addr)) {
            std::cout << "[CYCLE_DEBUG] Already processed: " << op->getName().getStringRef().str() << " (addr: " << addr << ")" << std::endl;
            return false;
        }

        // Add to current path
        std::cout << "[CYCLE_DEBUG] Adding to recursion stack: " << op->getName().getStringRef().str() << " (addr: " << addr << ")" << std::endl;
        recursionStack.insert(addr);

        // Manually traverse all nested operations
        for (auto& region : op->getRegions()) {
            std::cout << "[CYCLE_DEBUG] Processing region in " << op->getName().getStringRef().str() << std::endl;
            for (auto& block : region.getBlocks()) {
                std::cout << "[CYCLE_DEBUG] Processing block with " << block.getOperations().size() << " operations" << std::endl;
                for (auto& nestedOp : block.getOperations()) {
                    std::cout << "[CYCLE_DEBUG] About to recurse into: " << nestedOp.getName().getStringRef().str()
                              << " (addr: " << reinterpret_cast<uintptr_t>(&nestedOp) << ")" << std::endl;
                    if (detectCycle(&nestedOp)) {
                        return true;  // Cycle found in child
                    }
                }
                std::cout << "[CYCLE_DEBUG] Finished block with " << block.getOperations().size() << " operations" << std::endl;
            }
            std::cout << "[CYCLE_DEBUG] Finished " << op->getName().getStringRef().str() << std::endl;
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