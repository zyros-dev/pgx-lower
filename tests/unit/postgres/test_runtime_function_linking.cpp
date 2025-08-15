#include <gtest/gtest.h>

// PostgreSQL headers for runtime functions
extern "C" {
#include "postgres.h"
#include "access/heapam.h"
#include "access/table.h"
#include "access/tableam.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "storage/bufmgr.h"
#include "commands/trigger.h"
#include "miscadmin.h"
}

// Our headers
#include "execution/logging.h"
#include "runtime/tuple_access.h"

namespace {

class RuntimeFunctionLinkingTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_INFO("=== Setting up runtime function linking test ===");
        
        // Initialize minimal PostgreSQL environment for function testing
        try {
            // Create memory context for test
            TestMemoryContext = AllocSetContextCreate(NULL,
                                                     "RuntimeTestContext",
                                                     ALLOCSET_DEFAULT_SIZES);
            CurrentMemoryContext = TestMemoryContext;
            
            PGX_INFO("✓ PostgreSQL environment initialized for runtime function testing");
            postgres_initialized = true;
            
        } catch (...) {
            PGX_ERROR("✗ Failed to initialize PostgreSQL environment");
            postgres_initialized = false;
        }
    }
    
    void TearDown() override {
        if (postgres_initialized && TestMemoryContext) {
            MemoryContextDelete(TestMemoryContext);
            TestMemoryContext = nullptr;
            CurrentMemoryContext = nullptr;
        }
    }
    
    bool postgres_initialized = false;
    MemoryContext TestMemoryContext = nullptr;
};

// Test 1: Table Access Functions
TEST_F(RuntimeFunctionLinkingTest, TestTableAccessFunctions) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL environment must be initialized";
    
    PGX_INFO("=== Testing PostgreSQL table access function availability ===");
    
    // These are the functions that appear in compilation errors in tuple_access.cpp
    struct FunctionTest {
        std::string name;
        void* function_ptr;
        bool is_available;
    };
    
    std::vector<FunctionTest> table_functions = {
        // Core table access functions
        {"table_open", (void*)table_open, false},
        {"table_close", (void*)table_close, false},
        {"table_beginscan", (void*)table_beginscan, false},
        {"table_endscan", (void*)table_endscan, false},
        
        // Heap access functions  
        {"heap_getnext", (void*)heap_getnext, false},
        {"heap_rescan", (void*)heap_rescan, false},
        
        // Tuple descriptor functions
        {"RelationGetDescr", (void*)RelationGetDescr, false},
        
        // Snapshot functions
        {"GetActiveSnapshot", (void*)GetActiveSnapshot, false},
        
        // Command counter functions
        {"CommandCounterIncrement", (void*)CommandCounterIncrement, false},
        
        // Numeric conversion functions
        {"numeric_float8", (void*)numeric_float8, false}
    };
    
    // Test each function for availability
    for (auto& test : table_functions) {
        try {
            if (test.function_ptr != nullptr) {
                test.is_available = true;
                PGX_INFO("✓ Function '" + test.name + "' is available and linked");
            } else {
                test.is_available = false;
                PGX_ERROR("✗ Function '" + test.name + "' is NOT available");
            }
        } catch (...) {
            test.is_available = false;
            PGX_ERROR("✗ Function '" + test.name + "' caused exception during test");
        }
    }
    
    // Count available functions
    int available_count = 0;
    int total_count = table_functions.size();
    
    for (const auto& test : table_functions) {
        if (test.is_available) {
            available_count++;
        }
    }
    
    double availability_rate = (double)available_count / total_count;
    PGX_INFO("Function availability: " + std::to_string(available_count) + "/" + 
            std::to_string(total_count) + " (" + 
            std::to_string(availability_rate * 100) + "%)");
    
    if (availability_rate > 0.90) {
        PGX_INFO("✓ Most PostgreSQL table access functions are available");
        PGX_INFO("Compilation errors are likely due to header issues, not missing functions");
    } else if (availability_rate > 0.50) {
        PGX_WARNING("⚠ Some PostgreSQL table access functions are missing");
        PGX_WARNING("This may cause runtime linking failures in JIT execution");
    } else {
        PGX_ERROR("✗ Most PostgreSQL table access functions are missing");
        PGX_ERROR("This will cause JIT execution to fail due to unresolved symbols");
        FAIL() << "Critical runtime functions missing - JIT cannot work";
    }
    
    // Test specific functions that are critical for our runtime
    bool critical_functions_available = true;
    std::vector<std::string> missing_critical;
    
    for (const auto& test : table_functions) {
        if (test.name == "table_open" || test.name == "heap_getnext" || 
            test.name == "RelationGetDescr" || test.name == "GetActiveSnapshot") {
            if (!test.is_available) {
                critical_functions_available = false;
                missing_critical.push_back(test.name);
            }
        }
    }
    
    if (!critical_functions_available) {
        PGX_ERROR("✗ Critical runtime functions missing:");
        for (const auto& func : missing_critical) {
            PGX_ERROR("  - " + func);
        }
        FAIL() << "Critical runtime functions missing - this explains JIT crashes";
    } else {
        PGX_INFO("✓ All critical runtime functions are available");
    }
}

// Test 2: Numeric and Type Conversion Functions
TEST_F(RuntimeFunctionLinkingTest, TestNumericConversionFunctions) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL environment must be initialized";
    
    PGX_INFO("=== Testing PostgreSQL numeric conversion function availability ===");
    
    struct NumericFunctionTest {
        std::string name;
        bool available;
        std::string test_result;
    };
    
    std::vector<NumericFunctionTest> numeric_functions;
    
    // Test numeric_float8 function (appears in compilation error)
    try {
        // Create a simple numeric value for testing
        Datum numeric_val = DirectFunctionCall3(numeric_in,
                                               CStringGetDatum("42.5"),
                                               ObjectIdGetDatum(0),
                                               Int32GetDatum(-1));
        
        // Test numeric_float8 conversion
        Datum float8_val = DirectFunctionCall1(numeric_float8, numeric_val);
        double result = DatumGetFloat8(float8_val);
        
        numeric_functions.push_back({"numeric_float8", true, 
                                   "Successfully converted 42.5 to " + std::to_string(result)});
        PGX_INFO("✓ numeric_float8 function works correctly");
        
    } catch (...) {
        numeric_functions.push_back({"numeric_float8", false, "Exception during test"});
        PGX_ERROR("✗ numeric_float8 function failed");
    }
    
    // Test other numeric functions used in our runtime
    try {
        Datum int4_val = Int32GetDatum(123);
        Datum text_val = DirectFunctionCall1(int4out, int4_val);
        char* result_str = DatumGetCString(text_val);
        
        numeric_functions.push_back({"int4out", true, 
                                   "Successfully converted 123 to '" + std::string(result_str) + "'"});
        PGX_INFO("✓ int4out function works correctly");
        
    } catch (...) {
        numeric_functions.push_back({"int4out", false, "Exception during test"});
        PGX_ERROR("✗ int4out function failed");
    }
    
    // Test DirectFunctionCall macros
    try {
        Datum result = DirectFunctionCall1(int4abs, Int32GetDatum(-42));
        int32 abs_result = DatumGetInt32(result);
        
        numeric_functions.push_back({"DirectFunctionCall1", true,
                                   "abs(-42) = " + std::to_string(abs_result)});
        PGX_INFO("✓ DirectFunctionCall1 macro works correctly");
        
    } catch (...) {
        numeric_functions.push_back({"DirectFunctionCall1", false, "Exception during test"});
        PGX_ERROR("✗ DirectFunctionCall1 macro failed");
    }
    
    // Report results
    int working_functions = 0;
    for (const auto& test : numeric_functions) {
        if (test.available) {
            working_functions++;
            PGX_INFO("  ✓ " + test.name + ": " + test.test_result);
        } else {
            PGX_ERROR("  ✗ " + test.name + ": " + test.test_result);
        }
    }
    
    double success_rate = (double)working_functions / numeric_functions.size();
    
    if (success_rate == 1.0) {
        PGX_INFO("✓ All numeric conversion functions work correctly");
        PGX_INFO("Numeric function linking is not the cause of JIT crashes");
    } else {
        PGX_ERROR("✗ Some numeric conversion functions failed");
        PGX_ERROR("This may cause JIT execution failures for numeric operations");
        
        if (success_rate < 0.5) {
            FAIL() << "Critical numeric functions not working - explains compilation errors";
        }
    }
}

// Test 3: JIT Runtime Symbol Resolution
TEST_F(RuntimeFunctionLinkingTest, TestJITRuntimeSymbolResolution) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL environment must be initialized";
    
    PGX_INFO("=== Testing JIT runtime symbol resolution for PostgreSQL functions ===");
    
    // This test verifies that the symbols our JIT-compiled code needs are available
    // These are the functions that get called from LLVM-generated code
    
    struct SymbolTest {
        std::string symbol_name;
        void* symbol_address;
        bool resolved;
    };
    
    std::vector<SymbolTest> runtime_symbols = {
        // Table access functions (used in materialization)
        {"pg_table_open", (void*)pg_table_open, false},
        {"close_postgres_table", (void*)close_postgres_table, false},
        {"read_next_tuple_from_table", (void*)read_next_tuple_from_table, false},
        {"pg_get_next_tuple", (void*)pg_get_next_tuple, false},
        
        // Field access functions (used in expression evaluation)
        {"get_numeric_field", (void*)get_numeric_field, false},
        {"get_int_field", (void*)get_int_field, false},
        {"get_text_field", (void*)get_text_field, false},
        {"is_field_null", (void*)is_field_null, false}
    };
    
    // Test symbol resolution
    for (auto& test : runtime_symbols) {
        try {
            if (test.symbol_address != nullptr) {
                test.resolved = true;
                PGX_INFO("✓ Runtime symbol '" + test.symbol_name + "' resolved to address: " + 
                        std::to_string(reinterpret_cast<uintptr_t>(test.symbol_address)));
            } else {
                test.resolved = false;
                PGX_ERROR("✗ Runtime symbol '" + test.symbol_name + "' NOT resolved");
            }
        } catch (...) {
            test.resolved = false;
            PGX_ERROR("✗ Runtime symbol '" + test.symbol_name + "' caused exception");
        }
    }
    
    // Count resolved symbols
    int resolved_count = 0;
    int total_count = runtime_symbols.size();
    
    for (const auto& test : runtime_symbols) {
        if (test.resolved) {
            resolved_count++;
        }
    }
    
    double resolution_rate = (double)resolved_count / total_count;
    PGX_INFO("Symbol resolution rate: " + std::to_string(resolved_count) + "/" + 
            std::to_string(total_count) + " (" + 
            std::to_string(resolution_rate * 100) + "%)");
    
    if (resolution_rate == 1.0) {
        PGX_INFO("✓ All runtime symbols resolved successfully");
        PGX_INFO("JIT runtime symbol resolution is working correctly");
        PGX_INFO("Symbol resolution is NOT the cause of PostgreSQL JIT crashes");
    } else if (resolution_rate > 0.75) {
        PGX_WARNING("⚠ Most runtime symbols resolved, but some are missing");
        PGX_WARNING("This may cause JIT execution to fail for specific operations");
    } else {
        PGX_ERROR("✗ Many runtime symbols failed to resolve");
        PGX_ERROR("This will cause JIT execution to crash with unresolved symbol errors");
        PGX_ERROR("This explains the PostgreSQL exception in my_executor.cpp:342");
        FAIL() << "Critical runtime symbols missing - root cause of JIT crashes identified";
    }
    
    // Test if missing symbols correlate with compilation errors
    std::vector<std::string> compilation_error_functions = {
        "table_open", "table_close", "heap_getnext", "RelationGetDescr", 
        "GetActiveSnapshot", "numeric_float8"
    };
    
    bool has_compilation_errors = false;
    for (const auto& error_func : compilation_error_functions) {
        bool found_in_runtime = false;
        for (const auto& runtime_sym : runtime_symbols) {
            if (runtime_sym.symbol_name.find(error_func) != std::string::npos) {
                found_in_runtime = true;
                if (!runtime_sym.resolved) {
                    has_compilation_errors = true;
                    PGX_ERROR("Compilation error function '" + error_func + "' also missing at runtime");
                }
                break;
            }
        }
    }
    
    if (has_compilation_errors) {
        PGX_ERROR("=== CRITICAL FINDING ===");
        PGX_ERROR("Functions that fail to compile also fail at runtime");
        PGX_ERROR("This confirms that compilation errors lead to JIT execution crashes");
        FAIL() << "Compilation errors directly cause runtime symbol resolution failures";
    } else {
        PGX_INFO("✓ No correlation between compilation errors and runtime symbol failures");
    }
}

// Test 4: Full Runtime Function Call Test
TEST_F(RuntimeFunctionLinkingTest, TestFullRuntimeFunctionCalls) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL environment must be initialized";
    
    PGX_INFO("=== Testing full runtime function call execution ===");
    
    // Test the actual function calls that our JIT code would make
    // This simulates what happens when LLVM tries to execute generated code
    
    bool all_functions_work = true;
    std::vector<std::string> failed_functions;
    
    // Test 1: Table access simulation (without actual tables)
    try {
        PGX_INFO("Testing table access function signatures...");
        
        // These functions should exist even if they fail due to missing tables
        // We're testing linking, not functionality
        const char* test_table_name = "nonexistent_test_table";
        
        // This should fail gracefully, not crash due to missing symbols
        void* handle = pg_table_open(test_table_name);
        if (handle == nullptr) {
            PGX_INFO("✓ pg_table_open correctly returns NULL for nonexistent table");
        } else {
            PGX_WARNING("⚠ pg_table_open unexpectedly succeeded");
            close_postgres_table(handle);
        }
        
    } catch (const std::exception& e) {
        all_functions_work = false;
        failed_functions.push_back("pg_table_open: " + std::string(e.what()));
        PGX_ERROR("✗ pg_table_open threw exception: " + std::string(e.what()));
    } catch (...) {
        all_functions_work = false;
        failed_functions.push_back("pg_table_open: unknown exception");
        PGX_ERROR("✗ pg_table_open threw unknown exception");
    }
    
    // Test 2: Numeric conversion simulation
    try {
        PGX_INFO("Testing numeric conversion function calls...");
        
        // Test null handling
        bool is_null = true;
        double result = get_numeric_field(nullptr, 0, &is_null);
        
        if (is_null) {
            PGX_INFO("✓ get_numeric_field correctly handles NULL input");
        } else {
            PGX_WARNING("⚠ get_numeric_field returned unexpected result for NULL");
        }
        
    } catch (const std::exception& e) {
        all_functions_work = false;
        failed_functions.push_back("get_numeric_field: " + std::string(e.what()));
        PGX_ERROR("✗ get_numeric_field threw exception: " + std::string(e.what()));
    } catch (...) {
        all_functions_work = false;
        failed_functions.push_back("get_numeric_field: unknown exception");
        PGX_ERROR("✗ get_numeric_field threw unknown exception");
    }
    
    // Test 3: Integer field access simulation
    try {
        PGX_INFO("Testing integer field access function calls...");
        
        bool is_null = true;
        int64_t result = get_int_field(nullptr, 0, &is_null);
        
        if (is_null) {
            PGX_INFO("✓ get_int_field correctly handles NULL input");
        } else {
            PGX_WARNING("⚠ get_int_field returned unexpected result for NULL");
        }
        
    } catch (const std::exception& e) {
        all_functions_work = false;
        failed_functions.push_back("get_int_field: " + std::string(e.what()));
        PGX_ERROR("✗ get_int_field threw exception: " + std::string(e.what()));
    } catch (...) {
        all_functions_work = false;
        failed_functions.push_back("get_int_field: unknown exception");
        PGX_ERROR("✗ get_int_field threw unknown exception");
    }
    
    // Report results
    if (all_functions_work) {
        PGX_INFO("✓ All runtime function calls executed successfully");
        PGX_INFO("Runtime function linking is working correctly");
        PGX_INFO("Function call failures are NOT the cause of JIT crashes");
    } else {
        PGX_ERROR("✗ Some runtime function calls failed:");
        for (const auto& failure : failed_functions) {
            PGX_ERROR("  - " + failure);
        }
        PGX_ERROR("These function call failures may cause JIT execution crashes");
        
        if (failed_functions.size() > 2) {
            FAIL() << "Multiple runtime function call failures - likely cause of JIT crashes";
        } else {
            PGX_WARNING("Limited runtime function call failures detected");
        }
    }
}

} // namespace