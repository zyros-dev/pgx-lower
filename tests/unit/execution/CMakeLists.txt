add_executable(test_executor_helpers test_executor_helpers.cpp)

target_link_libraries(test_executor_helpers
    gtest 
    gtest_main
)

target_include_directories(test_executor_helpers PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src
)

# Explicitly disable PostgreSQL extension for unit tests
target_compile_definitions(test_executor_helpers PRIVATE POSTGRESQL_EXTENSION=0)

add_test(NAME ExecutorHelpersTest COMMAND test_executor_helpers)

# Test for WrappedExecutionEngine enhancements
add_executable(test_wrapped_execution_engine test_wrapped_execution_engine.cpp)

target_link_libraries(test_wrapped_execution_engine
    gtest 
    gtest_main
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
)

target_include_directories(test_wrapped_execution_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

add_test(NAME WrappedExecutionEngineTest COMMAND test_wrapped_execution_engine)