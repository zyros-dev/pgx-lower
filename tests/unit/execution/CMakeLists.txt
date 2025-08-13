# Execution unit tests
add_executable(test_phase3b_validation test_phase3b_validation.cpp)
target_link_libraries(test_phase3b_validation
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRUtilDialect
    MLIRFuncDialect
    MLIRPass
    pgx_lower_passes
    pgx_lower_logging
)
add_test(NAME Phase3bValidationTest COMMAND test_phase3b_validation)

# PostgreSQL integration safety tests
add_executable(test_postgresql_integration_safety test_postgresql_integration_safety.cpp)
target_link_libraries(test_postgresql_integration_safety
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRUtilDialect
    MLIRFuncDialect
    MLIRPass
    pgx_lower_logging
)
add_test(NAME PostgreSQLIntegrationSafetyTest COMMAND test_postgresql_integration_safety)