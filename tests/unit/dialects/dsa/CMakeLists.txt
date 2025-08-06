# DSA Dialect Unit Tests CMakeLists.txt

add_executable(test_dsa_basic test_dsa_basic.cpp)
target_link_libraries(test_dsa_basic
    gtest gtest_main
    MLIRDSADialect
    MLIRIR
    MLIRArithDialect
    MLIRTestDialect
)
add_test(NAME DSABasicTest COMMAND test_dsa_basic)