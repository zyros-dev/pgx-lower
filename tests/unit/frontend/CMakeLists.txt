# Frontend unit tests

# Test for plan node type translation
add_executable(test_plan_node_types test_plan_node_types.cpp
    ${CMAKE_SOURCE_DIR}/src/frontend/SQL/postgresql_ast_translator.cpp
)

target_include_directories(test_plan_node_types PRIVATE
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include/pgx_lower
    ${CMAKE_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(test_plan_node_types
    gtest
    gtest_main
    pgx_lower_logging
    pgx_lower_runtime
    RelAlgDialect
    DSADialect
    UtilDialect
    DBDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRTransforms
)

add_test(NAME PlanNodeTypesTest COMMAND test_plan_node_types)