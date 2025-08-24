find_package(GTest REQUIRED)

add_executable(test_mlir_lowerings
    standalone_mlir_runner.cpp
    test_pipeline_phases.cpp

    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_runner.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp

    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_tools.cpp

    ${CMAKE_SOURCE_DIR}/src/lingodb/runtime/MetaData.cpp
)

add_executable(test_boolean_lowering
    test_boolean_lowering.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_tools.cpp
)

# Set the compile definition to enable standalone mode
target_compile_definitions(test_mlir_lowerings PRIVATE 
    BUILDING_UNIT_TESTS=1
)

target_include_directories(test_mlir_lowerings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lingo-db/include
    ${CMAKE_SOURCE_DIR}/src
)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

set(MLIR_CORE_LIBS
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIROptLib
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRExecutionEngine
)

set(MLIR_ALL_LIBS
    /usr/lib/llvm-20/lib/libMLIR.so.20.1
)

target_link_libraries(test_mlir_lowerings
    gtest
    gtest_main
    
    ${MLIR_ALL_LIBS}
    
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    MLIRRelAlgToDB
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRUtilToLLVM
    MLIRPasses
    MLIRCustomTransforms
    
    ${LLVM_LIBRARIES}
)

llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(test_mlir_lowerings ${llvm_libs})

target_include_directories(test_mlir_lowerings PRIVATE ${MLIR_INCLUDE_DIRS})
target_include_directories(test_mlir_lowerings PRIVATE ${LLVM_INCLUDE_DIRS})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(test_mlir_lowerings PRIVATE ${LLVM_DEFINITIONS_LIST})

target_compile_features(test_mlir_lowerings PRIVATE cxx_std_20)

add_test(NAME MLIRLoweringTests COMMAND test_mlir_lowerings)

set_tests_properties(MLIRLoweringTests PROPERTIES
    TIMEOUT 600
    LABELS "unit;mlir;lowering"
)

# Configure boolean lowering test
target_compile_definitions(test_boolean_lowering PRIVATE 
    BUILDING_UNIT_TESTS=1
)

target_include_directories(test_boolean_lowering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lingo-db/include
    ${CMAKE_SOURCE_DIR}/src
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(test_boolean_lowering
    gtest
    gtest_main
    ${MLIR_ALL_LIBS}
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRUtilToLLVM
    ${llvm_libs}
)

target_compile_features(test_boolean_lowering PRIVATE cxx_std_20)
target_compile_definitions(test_boolean_lowering PRIVATE ${LLVM_DEFINITIONS_LIST})

add_test(NAME BooleanLoweringTest COMMAND test_boolean_lowering)

set_tests_properties(BooleanLoweringTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;mlir;lowering;boolean"
)