# Extension Loading Unit Tests

# Test Extension Loading
add_executable(test_loading test_loading.cpp)
target_link_libraries(test_loading
    gtest gtest_main
    DSADialect
    RelAlgDialect
    DBDialect
    UtilDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRIR
    MLIRParser
    pgx_lower_common
)
add_test(NAME ExtensionLoadingTest COMMAND test_loading)

# Additional extension tests can be added here