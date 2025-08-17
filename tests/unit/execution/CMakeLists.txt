# JIT Execution Engine Tests

# Test for JIT execution engine
add_executable(test_jit_execution_engine test_jit_execution_engine.cpp)

target_link_libraries(test_jit_execution_engine
    gtest 
    gtest_main
    pgx_lower_jit  # Use the existing JIT library from tests/unit/mlir
    ${MLIR_LIBRARIES}
    ${LLVM_LIBRARIES}
)

target_include_directories(test_jit_execution_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

add_test(NAME JITExecutionEngineTest COMMAND test_jit_execution_engine)