// Example usage of the unified logging system
// This file demonstrates how to use the new logging macros

#include "core/logging.h"
#include "core/mlir_logger.h"
#include <string>

void example_unified_logging() {
    // Basic PGX logging (existing functionality)
    PGX_DEBUG("Debug message using PGX logging");
    PGX_INFO("Info message using PGX logging");
    PGX_ERROR("Error message using PGX logging");
    PGX_NOTICE("Notice message using PGX logging");
    
    // MLIR-specific logging with dialect context
    MLIR_PGX_DEBUG("DB", "Debug message in DB dialect lowering");
    MLIR_PGX_INFO("SubOp", "Info message in SubOp dialect processing");
    MLIR_PGX_ERROR("RelAlg", "Error message in RelAlg dialect conversion");
    
    // Runtime-specific logging with component context
    RUNTIME_PGX_DEBUG("TupleAccess", "Debug message in tuple access runtime");
    RUNTIME_PGX_NOTICE("PostgreSQL", "Notice message in PostgreSQL runtime");
}

void example_mlir_logger_usage() {
    // Using the MLIRLogger interface directly
    ConsoleLogger console_logger;
    PostgreSQLLogger pg_logger;
    
    console_logger.debug("Direct debug message to console");
    console_logger.notice("Direct notice message to console");
    console_logger.error("Direct error message to console");
    
    // PostgreSQL logger would use elog() internally
    // (only use in PostgreSQL extension context)
    // pg_logger.debug("Direct debug message via PostgreSQL elog");
}

// Replacement patterns for existing code:
//
// OLD: elog(NOTICE, "message")
// NEW: PGX_NOTICE("message")
//
// OLD: elog(DEBUG1, "message") 
// NEW: PGX_DEBUG("message")
//
// OLD: errs() << "message"
// NEW: PGX_DEBUG("message")
//
// OLD: llvm::errs() << "message"
// NEW: PGX_DEBUG("message")
//
// MLIR dialect-specific:
// NEW: MLIR_PGX_DEBUG("DB", "message in DB dialect")
// NEW: MLIR_PGX_INFO("SubOp", "message in SubOp dialect")
//
// Runtime component-specific:
// NEW: RUNTIME_PGX_DEBUG("TupleAccess", "message in tuple access")
// NEW: RUNTIME_PGX_NOTICE("PostgreSQL", "message in PostgreSQL runtime")