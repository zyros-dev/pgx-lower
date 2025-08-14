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

# Phase execution tests
add_executable(test_phase_execution test_phase_execution.cpp)
target_link_libraries(test_phase_execution
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRUtilDialect
    MLIRFuncDialect
    MLIRPass
    pgx_lower_passes
    pgx_lower_logging
    # No need to link mlir_runner since we're mocking the functions
)
add_test(NAME PhaseExecutionTest COMMAND test_phase_execution)

# Module validation tests
add_executable(test_module_validation test_module_validation.cpp)
target_link_libraries(test_module_validation
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRUtilDialect
    MLIRFuncDialect
    MLIRPass
    pgx_lower_logging
    # No need to link mlir_runner since we're mocking the functions
)
add_test(NAME ModuleValidationTest COMMAND test_module_validation)