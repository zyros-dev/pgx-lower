# Frontend unit tests

# Simple test to verify CMake configuration
add_executable(test_simple test_simple.cpp)
target_link_libraries(test_simple gtest gtest_main)
add_test(NAME SimpleTest COMMAND test_simple)

# Test for plan node type translation - minimal test
add_executable(test_plan_node_types_simple 
    test_plan_node_types.cpp
    ${CMAKE_SOURCE_DIR}/src/frontend/SQL/postgresql_ast_translator.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/MetaData.cpp
)

target_include_directories(test_plan_node_types_simple PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(test_plan_node_types_simple
    gtest
    gtest_main
    pgx_lower_logging
    MLIRRelAlgDialect
    MLIRDSA
    MLIRUtilDialect
    MLIRDBDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRTransforms
)

add_test(NAME PlanNodeTypesSimpleTest COMMAND test_plan_node_types_simple)