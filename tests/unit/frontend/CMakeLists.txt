# Frontend unit tests

# PostgreSQL AST Translator test
add_executable(test_postgresql_ast_translator test_postgresql_ast_translator.cpp)
target_link_libraries(test_postgresql_ast_translator
    gtest gtest_main
    RelAlgDialect
    MLIRRelAlgOps
    MLIRFuncDialect
    MLIRSupport
    MLIRIR
    pgx_lower
)
target_include_directories(test_postgresql_ast_translator PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${PostgreSQL_INCLUDE_DIRS}
)
add_test(NAME PostgreSQLASTTranslatorTest COMMAND test_postgresql_ast_translator)