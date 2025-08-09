# Execution Unit Tests CMakeLists.txt

add_executable(test_jit_execution_engine test_jit_execution_engine.cpp)
target_link_libraries(test_jit_execution_engine
    gtest gtest_main
    pgx_lower_lib
    MLIRExecutionEngine
    MLIRIR
    MLIRFuncDialect
    MLIRLLVMDialect
)
add_test(NAME JITExecutionEngineTest COMMAND test_jit_execution_engine)

add_executable(test_runtime_symbol_registration test_runtime_symbol_registration.cpp)
target_link_libraries(test_runtime_symbol_registration
    gtest gtest_main
    pgx_lower_lib
    MLIRExecutionEngine
    MLIRIR
)
add_test(NAME RuntimeSymbolRegistrationTest COMMAND test_runtime_symbol_registration)

add_executable(test_jit_execution_integration test_jit_execution_integration.cpp)
target_link_libraries(test_jit_execution_integration
    gtest gtest_main
    pgx_lower_lib
    MLIRExecutionEngine
    MLIRIR
    MLIRFuncDialect
    MLIRLLVMDialect
)
add_test(NAME JITExecutionIntegrationTest COMMAND test_jit_execution_integration)

add_executable(test_function_decomposition test_function_decomposition.cpp)
target_link_libraries(test_function_decomposition
    gtest gtest_main
    pgx_lower_lib
    MLIRIR
)
add_test(NAME FunctionDecompositionTest COMMAND test_function_decomposition)