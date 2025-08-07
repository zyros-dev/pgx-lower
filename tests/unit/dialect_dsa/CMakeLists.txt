# Unit tests for DSA dialect operations
# Validates new LingoDB-compliant DSA operations

add_executable(test_dsa_operations test_dsa_operations.cpp)

target_link_libraries(test_dsa_operations
    gtest 
    gtest_main
    MLIRDSADialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register DSA operations tests with CTest
add_test(NAME DSAOperationsTest COMMAND test_dsa_operations)