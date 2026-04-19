find_package(GTest REQUIRED)

# test_mlir_lowerings is currently unbuildable without a real refactor.
#
# Problem: test_pipeline_phases.cpp + mlir_runner.cpp + mlir_runner_phases.cpp
# transitively drag in source that uses PG backend symbols (pg_fprintf,
# CHECK_FOR_INTERRUPTS, GetDefaultOpClass / get_opclass_family /
# get_opfamily_member, the Numeric type, the JITEngine full impl that
# depends on tuple_access.h) which don't exist when you link outside a
# running PG. Adding stub .cpp files covers the catalog helpers but the
# cascade keeps going — MLIR forward-decls in logging.h aren't enough,
# tuple_access.h unconditionally references Numeric, etc.
#
# Short fix: enable ENABLE_BROKEN_MLIR_LOWERING_TESTS=ON to try to build
# it anyway (for whoever's refactoring). Real fix: split mlir_to_postgres.cpp
# and the rest of the cascade into a catalog-free shim + catalog-using
# impl and link only the shim into tests. Tracked as harness work;
# doesn't block general unit-test TDD (see test_boolean_lowering below).
option(ENABLE_BROKEN_MLIR_LOWERING_TESTS "Try to build test_mlir_lowerings (currently unbuildable; needs source refactor)" OFF)

if(ENABLE_BROKEN_MLIR_LOWERING_TESTS)
add_executable(test_mlir_lowerings
    standalone_mlir_runner.cpp
    test_pipeline_phases.cpp
    pg_stubs.cpp

    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_runner.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/mlir_logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/src/lingodb/runtime/MetaData.cpp
)
endif()

add_executable(test_boolean_lowering
    test_boolean_lowering.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

if(ENABLE_BROKEN_MLIR_LOWERING_TESTS)

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

endif()  # ENABLE_BROKEN_MLIR_LOWERING_TESTS

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