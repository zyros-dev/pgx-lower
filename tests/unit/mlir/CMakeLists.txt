if(NOT TARGET pgx_lower_logging)
    add_library(pgx_lower_logging STATIC
        ${CMAKE_SOURCE_DIR}/src/utility/logging.cpp
        ${CMAKE_SOURCE_DIR}/src/utility/logging_tools.cpp
        ${CMAKE_SOURCE_DIR}/src/utility/logging_c.cpp
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
        ${PostgreSQL_INCLUDE_DIRS}
        ${PostgreSQL_SERVER_INCLUDE_DIRS}
    )
    target_link_libraries(pgx_lower_logging PUBLIC
        ${PostgreSQL_LIBRARIES}
    )
endif()

if(NOT TARGET pgx_lower_jit)
    add_library(pgx_lower_jit STATIC
        ${CMAKE_SOURCE_DIR}/src/execution/jit_execution_engine.cpp
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

if(NOT TARGET pgx_lower_passes)
    add_library(pgx_lower_passes INTERFACE)
    target_link_libraries(pgx_lower_passes INTERFACE
        MLIRPasses
        StandardToLLVM
        MLIRDBToArrowStd
        MLIRDSAToStd
        MLIRUtilToLLVM
        MLIRRelAlgToDB
        MLIRSCFToControlFlow
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRReconcileUnrealizedCasts
    )
endif()

# Test for dialect verification logic
add_executable(test_dialect_verification 
    test_dialect_verification.cpp
)

target_include_directories(test_dialect_verification PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

target_link_libraries(test_dialect_verification
    gtest gtest_main
    pgx_lower_logging
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRIR
)

add_test(NAME DialectVerificationTest COMMAND test_dialect_verification)

# Simpler test for LLVM passes debugging
add_executable(test_llvm_passes_simple 
    test_llvm_passes_simple.cpp
)

target_include_directories(test_llvm_passes_simple PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(test_llvm_passes_simple
    gtest gtest_main
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRControlFlowDialect
    MLIRPass
    MLIRIR
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
    pgx_lower_logging
)

add_test(NAME LLVMPassesSimpleTest COMMAND test_llvm_passes_simple)

# Create mlir_unit_test executable that runs all unit tests
add_executable(mlir_unit_test
    test_dialect_verification.cpp
    test_llvm_passes_simple.cpp
    test_real_module_crash.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/MetaData.cpp
)

target_include_directories(mlir_unit_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/mlir/Conversion
    ${CMAKE_SOURCE_DIR}/src/execution
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/pgx_lower
)

target_link_libraries(mlir_unit_test
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    StandardToLLVM
    pgx_lower_passes
    MLIRPasses  # Contains createRelAlgToDBPipeline for crash test
    pgx_lower_logging
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