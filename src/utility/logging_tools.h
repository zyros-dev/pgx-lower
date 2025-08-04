#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

// Forward declarations
class MLIRLogger;

namespace pgx_lower {
namespace utility {

/**
 * @brief Log MLIR module with verbose operation-by-operation breakdown
 * 
 * This function walks through all operations in the module and logs each one
 * individually with full details. Useful for debugging when operations contain
 * nested regions that get truncated in compact dumps (like subop.execution_group).
 * 
 * @param module The MLIR module to log
 * @param logger The logger instance to use
 * @param context Context string to include in log messages
 */
void logMLIRModuleVerbose(mlir::ModuleOp module, MLIRLogger& logger, const std::string& context);

/**
 * @brief Log MLIR module with full expansion of all regions and operations
 * 
 * This function forces complete expansion of all nested regions and operations
 * without any folding or truncation. Use this when you need to see the complete
 * structure of complex operations like subop.execution_group that contain nested regions.
 * 
 * @param module The MLIR module to log
 * @param logger The logger instance to use
 * @param context Context string to include in log messages
 */
void logMLIRModuleFullyExpanded(mlir::ModuleOp module, MLIRLogger& logger, const std::string& context);

/**
 * @brief Log MLIR module in compact format
 * 
 * This function prints the entire module as a single string and splits it into
 * lines for logging. More compact than verbose mode but may truncate nested regions.
 * 
 * @param module The MLIR module to log
 * @param logger The logger instance to use
 * @param context Context string to include in log messages
 */
void logMLIRModuleCompact(mlir::ModuleOp module, MLIRLogger& logger, const std::string& context);

/**
 * @brief Log a single MLIR operation with full details
 * 
 * @param op The MLIR operation to log
 * @param logger The logger instance to use
 * @param context Context string to include in log messages
 */
void logMLIROperation(mlir::Operation* op, MLIRLogger& logger, const std::string& context);

/**
 * @brief Log MLIR operation hierarchy with indentation
 * 
 * This function recursively logs an operation and all its nested operations
 * with proper indentation to show the hierarchy structure.
 * 
 * @param op The root MLIR operation to log
 * @param logger The logger instance to use
 * @param context Context string to include in log messages
 * @param depth Current nesting depth (for indentation)
 */
void logMLIROperationHierarchy(mlir::Operation* op, MLIRLogger& logger, const std::string& context, int depth = 0);

} // namespace utility
} // namespace pgx_lower