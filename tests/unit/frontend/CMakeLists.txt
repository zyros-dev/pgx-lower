# Frontend unit tests

# Simple test to verify CMake configuration
add_executable(test_simple test_simple.cpp)
target_link_libraries(test_simple gtest gtest_main)
add_test(NAME SimpleTest COMMAND test_simple)

# Common libraries for all frontend tests
set(FRONTEND_TEST_LIBS
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

# Common include directories for all frontend tests
set(FRONTEND_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}  # For test_plan_node_helpers.h
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

# Common source files needed by tests
set(FRONTEND_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/frontend/SQL/postgresql_ast_translator.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/MetaData.cpp
)

# Test for basic plan nodes (SeqScan, Agg, Sort, Limit, Gather)
add_executable(test_basic_plan_nodes 
    test_basic_plan_nodes.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_basic_plan_nodes PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_basic_plan_nodes PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_basic_plan_nodes ${FRONTEND_TEST_LIBS})
add_test(NAME BasicPlanNodesTest COMMAND test_basic_plan_nodes)

# Test for complex plan trees (multi-level trees, projections)
add_executable(test_complex_plan_trees
    test_complex_plan_trees.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_complex_plan_trees PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_complex_plan_trees PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_complex_plan_trees ${FRONTEND_TEST_LIBS})
add_test(NAME ComplexPlanTreesTest COMMAND test_complex_plan_trees)

# Test for arithmetic expressions
add_executable(test_arithmetic_expressions
    test_arithmetic_expressions.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_arithmetic_expressions PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_arithmetic_expressions PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_arithmetic_expressions ${FRONTEND_TEST_LIBS})
add_test(NAME ArithmeticExpressionsTest COMMAND test_arithmetic_expressions)

# Test for logical and comparison expressions
add_executable(test_logical_expressions
    test_logical_expressions.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_logical_expressions PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_logical_expressions PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_logical_expressions ${FRONTEND_TEST_LIBS})
add_test(NAME LogicalExpressionsTest COMMAND test_logical_expressions)

# Test for error handling and edge cases
add_executable(test_error_handling
    test_error_handling.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_error_handling PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_error_handling PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_error_handling ${FRONTEND_TEST_LIBS})
add_test(NAME ErrorHandlingTest COMMAND test_error_handling)

# Keep the original large test file for reference (but renamed)
add_executable(test_plan_node_types_full
    test_plan_node_types.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_plan_node_types_full PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_plan_node_types_full PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_plan_node_types_full ${FRONTEND_TEST_LIBS})
add_test(NAME PlanNodeTypesFullTest COMMAND test_plan_node_types_full)

# Test for BaseTableOp attribute printing fix
add_executable(test_base_table_op 
    test_base_table_op.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_base_table_op PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_base_table_op PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_base_table_op ${FRONTEND_TEST_LIBS})
add_test(NAME BaseTableOpTest COMMAND test_base_table_op)

# Test for AST translator with fixed PostgreSQL 17 structures
add_executable(test_ast_translator_fixed
    test_ast_translator_fixed.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_ast_translator_fixed PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_ast_translator_fixed PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_ast_translator_fixed ${FRONTEND_TEST_LIBS})
add_test(NAME ASTTranslatorFixedTest COMMAND test_ast_translator_fixed)

# Test for OpExpr translation functions
add_executable(test_opexpr_translation
    test_opexpr_translation.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_opexpr_translation PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_opexpr_translation PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_opexpr_translation ${FRONTEND_TEST_LIBS})
add_test(NAME OpExprTranslationTest COMMAND test_opexpr_translation)

# Test for multi-column basetable translation (reproduces Test 4 bug)
add_executable(test_multi_column_basetable
    test_multi_column_basetable.cpp
    ${FRONTEND_TEST_SOURCES}
)
target_include_directories(test_multi_column_basetable PRIVATE ${FRONTEND_TEST_INCLUDES})
target_compile_definitions(test_multi_column_basetable PRIVATE BUILDING_UNIT_TESTS)
target_link_libraries(test_multi_column_basetable ${FRONTEND_TEST_LIBS})
add_test(NAME MultiColumnBaseTableTest COMMAND test_multi_column_basetable)