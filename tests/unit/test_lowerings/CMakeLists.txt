find_package(GTest REQUIRED)

# test_mlir_lowerings links against code that, in the POSTGRESQL_EXTENSION
# build, calls into PG's backend (palloc machinery, catalog lookups,
# CHECK_FOR_INTERRUPTS, pg_fprintf). For the unit-test build those are
# either #ifdef-ed out at the call site (guarded by POSTGRESQL_EXTENSION)
# or satisfied by minimal stubs in pg_stubs.cpp below — MLIR-pipeline tests
# don't exercise those paths, so stubs that `abort()` or return a trivial
# default are enough.
#
# If a new transitive dep introduces more PG-symbol leaks and breaks the
# link, two knobs: extend pg_stubs.cpp (for leaf calls that never fire in
# tests), or add a POSTGRESQL_EXTENSION guard at the offending call site
# (for calls that would actually corrupt the test if run).
#
# Flip OFF only if you've broken the link and need the rest of the utest
# suite to keep running while you debug.
option(ENABLE_MLIR_LOWERING_TESTS "Build the test_mlir_lowerings gtest binary. Requires pg_stubs.cpp to cover any PG symbols leaked through transitive deps." ON)

if(ENABLE_MLIR_LOWERING_TESTS)
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

if(ENABLE_MLIR_LOWERING_TESTS)

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

endif()  # ENABLE_MLIR_LOWERING_TESTS

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

# ----------------------------------------------------------------------------
# Type-mapping smoke test — proves pure-computation functions in
# lingodb_utility are unit-testable despite the library's PG-heavy context.
# Also catches type-mapping contract changes.

add_executable(test_type_mapping
    test_type_mapping.cpp
    pg_stubs.cpp
)

target_compile_definitions(test_type_mapping PRIVATE
    BUILDING_UNIT_TESTS=1
    ${LLVM_DEFINITIONS_LIST}
)

target_include_directories(test_type_mapping PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lingo-db/include
    ${CMAKE_SOURCE_DIR}/src
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    /usr/local/pgsql/include/server
)

target_link_libraries(test_type_mapping
    gtest
    gtest_main
    ${MLIR_ALL_LIBS}
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRUtilToLLVM
    lingodb_utility
    ${llvm_libs}
)

target_compile_features(test_type_mapping PRIVATE cxx_std_20)

add_test(NAME TypeMappingTest COMMAND test_type_mapping)

set_tests_properties(TypeMappingTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;types;mapping"
)

# ----------------------------------------------------------------------------
# AST Const-translation smoke test — proves the frontend/SQL/translation
# layer is unit-testable. Tests the simpler branches of translate_const
# (integer/float/bool/null/date) that don't require PG runtime — the
# NUMERICOID and TIMESTAMPOID branches are already POSTGRESQL_EXTENSION-
# guarded in the source.

add_executable(test_ast_const_translation
    test_ast_const_translation.cpp
    pg_stubs.cpp

    # translate_const + PostgreSQLTypeMapper live here.
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/frontend/SQL/translation/translation_core.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
)

target_compile_definitions(test_ast_const_translation PRIVATE
    BUILDING_UNIT_TESTS=1
    ${LLVM_DEFINITIONS_LIST}
)

target_include_directories(test_ast_const_translation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lingo-db/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/frontend/SQL
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    /usr/local/pgsql/include/server
)

target_link_libraries(test_ast_const_translation
    gtest
    gtest_main
    ${MLIR_ALL_LIBS}
    MLIRDBDialect
    MLIRDSA
    MLIRUtilDialect
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRUtilToLLVM
    lingodb_utility
    ${llvm_libs}
)

target_compile_features(test_ast_const_translation PRIVATE cxx_std_20)

add_test(NAME AstConstTranslationTest COMMAND test_ast_const_translation)

set_tests_properties(AstConstTranslationTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;ast;translation"
)

# ----------------------------------------------------------------------------
# Version-function smoke test (spec 14) — exercises the utest harness
# for a trivial C function that returns a constant string. The live PG
# boundary (PG_RETURN_TEXT_P/cstring_to_text) can't run in a unit test,
# so we pin the constant itself.

add_executable(test_version_function
    test_version_function.cpp
)

target_compile_definitions(test_version_function PRIVATE
    BUILDING_UNIT_TESTS=1
)

target_link_libraries(test_version_function
    gtest
    gtest_main
)

target_compile_features(test_version_function PRIVATE cxx_std_20)

add_test(NAME VersionFunctionTest COMMAND test_version_function)

set_tests_properties(VersionFunctionTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;version"
)