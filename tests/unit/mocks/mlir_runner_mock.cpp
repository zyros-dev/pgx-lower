#include "execution/mlir_runner.h"
#include "../test_helpers.h"

// Mock implementations of mlir_runner functions for unit tests
// These are completely isolated from PostgreSQL

extern "C" {
    // Mock runtime functions
    int64_t get_int_field_mlir(int64_t tuple_ptr, int field_index, bool* is_null) {
        *is_null = false;
        return field_index * 42; // Mock data based on field index
    }
    
    void mark_results_ready_for_streaming() {
        // Mock implementation - does nothing
    }
    
    void store_field_as_datum(int64_t field_value, int field_index) {
        // Mock implementation - does nothing
    }
}

namespace mlir_runner {
    
    bool run_mlir_postgres_table_scan(const std::string& table_name, MLIRLogger& logger) {
        // Mock implementation always succeeds
        return true;
    }
    
    bool run_mlir_postgres_typed_table_scan(const std::string& table_name, MLIRLogger& logger) {
        // Mock implementation always succeeds
        return true;
    }
}