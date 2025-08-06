# DB Dialect Unit Tests CMakeLists.txt

add_executable(test_db_basic_simple test_db_basic_simple.cpp)
target_link_libraries(test_db_basic_simple
    gtest gtest_main
    MLIRDBDialect
    MLIRIR
    MLIRArithDialect
    MLIRTestDialect
)
add_test(NAME DBBasicSimpleTest COMMAND test_db_basic_simple)

add_executable(test_db_comprehensive test_db_comprehensive.cpp)
target_link_libraries(test_db_comprehensive
    gtest gtest_main
    MLIRDBDialect
    MLIRIR
    MLIRArithDialect
    MLIRTestDialect
)
add_test(NAME DBComprehensiveTest COMMAND test_db_comprehensive)