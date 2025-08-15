# Unit tests for sequential pipeline approach
add_executable(test_sequential_pipelines 
    test_sequential_pipelines.cpp
    runtime_stubs.cpp  # Add minimal runtime stubs
)

target_include_directories(test_sequential_pipelines PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

# Create a local logging library for unit tests if not already defined
if(NOT TARGET pgx_lower_logging)
    add_library(pgx_lower_logging STATIC
        ${CMAKE_SOURCE_DIR}/src/utility/logging.cpp
        ${CMAKE_SOURCE_DIR}/src/utility/logging_tools.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/error_handling.cpp
    )
    target_include_directories(pgx_lower_logging PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/pgx_lower
        ${CMAKE_SOURCE_DIR}/src/execution
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/RelAlg/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/DB/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/DSA/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/util/include
    )
endif()

# Create JIT execution library for unit tests
if(NOT TARGET pgx_lower_jit)
    add_library(pgx_lower_jit STATIC
        ${CMAKE_SOURCE_DIR}/src/execution/jit_execution_engine.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/jit_execution_wrapper.cpp
        ${CMAKE_SOURCE_DIR}/src/execution/mlir_runner.cpp
        ${CMAKE_SOURCE_DIR}/src/frontend/SQL/postgresql_ast_translator.cpp
        ${CMAKE_SOURCE_DIR}/src/frontend/SQL/query_analyzer.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/helpers.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/PostgreSQLDataSource.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/postgresql_runtime.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/tuple_access.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/MetaData.cpp
    )
    target_include_directories(pgx_lower_jit PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/pgx_lower
        ${CMAKE_SOURCE_DIR}/src/execution
        ${CMAKE_SOURCE_DIR}/src/frontend/SQL
        ${CMAKE_SOURCE_DIR}/src/runtime
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/RelAlg/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/DB/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/DSA/include
        ${CMAKE_SOURCE_DIR}/src/mlir/Dialect/util/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${PostgreSQL_SERVER_INCLUDE_DIRS}
    )
    target_link_libraries(pgx_lower_jit PUBLIC
        MLIRRelAlgDialect
        MLIRDBDialect
        MLIRDSA
        MLIRUtilDialect
        pgx_lower_passes
        pgx_lower_logging
        MLIRExecutionEngine
        MLIRJitRunner
        ${PostgreSQL_LIBRARIES}
    )
endif()

# Create a local passes library for unit tests if not already defined
if(NOT TARGET pgx_lower_passes)
    add_library(pgx_lower_passes INTERFACE)
    target_link_libraries(pgx_lower_passes INTERFACE
        MLIRPasses
        StandardToLLVM
        MLIRDBToArrowStd
        MLIRDSAToStd
        UtilToLLVM
        MLIRRelAlgToDB
        MLIRSCFToControlFlow
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRReconcileUnrealizedCasts
    )
endif()

target_link_libraries(test_sequential_pipelines
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    pgx_lower_passes
    pgx_lower_logging
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
)

add_test(NAME SequentialPipelinesTest COMMAND test_sequential_pipelines)

# Test for DSA to Standard lowering patterns
add_executable(test_dsa_to_std_lowering 
    test_dsa_to_std_lowering.cpp
    runtime_stubs.cpp
)

target_include_directories(test_dsa_to_std_lowering PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

target_link_libraries(test_dsa_to_std_lowering
    gtest gtest_main
    MLIRDSA
    MLIRUtilDialect
    MLIRDSAToStd
    MLIRRelAlgDialect  # Add for column management
    pgx_lower_passes
    pgx_lower_logging
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
)

add_test(NAME DSAToStdLoweringTest COMMAND test_dsa_to_std_lowering)

# Test for JIT execution debugging
add_executable(test_jit_execution_debug 
    test_jit_execution_debug.cpp
    runtime_stubs.cpp
)

target_include_directories(test_jit_execution_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

target_link_libraries(test_jit_execution_debug
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    pgx_lower_passes
    pgx_lower_logging
    pgx_lower_jit
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
)

add_test(NAME JITExecutionDebugTest COMMAND test_jit_execution_debug)

# Test for function preservation through lowering passes
add_executable(test_function_preservation 
    test_function_preservation.cpp
    runtime_stubs.cpp
)

target_include_directories(test_function_preservation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

target_link_libraries(test_function_preservation
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    pgx_lower_passes
    pgx_lower_logging
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
)

add_test(NAME FunctionPreservationTest COMMAND test_function_preservation)