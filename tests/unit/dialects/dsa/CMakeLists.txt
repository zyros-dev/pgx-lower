# DSA Dialect Unit Tests

# Test DSA Integration
add_executable(test_dsa_integration test_dsa_integration.cpp)
target_link_libraries(test_dsa_integration
    gtest gtest_main
    DSADialect
    RelAlgDialect
    DBDialect
    UtilDialect
    MLIRDSAToStd
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    pgx_lower_common
)
add_test(NAME DSAIntegrationTest COMMAND test_dsa_integration)

# Additional DSA-specific tests can be added here as needed