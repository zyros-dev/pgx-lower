# Unit tests for MLIR conversion passes

# Test for DB constant char type lowering
add_executable(test_db_constant_char test_db_constant_char.cpp)
target_link_libraries(test_db_constant_char
    gtest gtest_main
    MLIRDBDialect
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
)
add_test(NAME DBConstantCharTest COMMAND test_db_constant_char)

# Test for DSA char append handling
add_executable(test_dsa_char_append 
    test_dsa_char_append.cpp
    ${CMAKE_SOURCE_DIR}/src/lingodb/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/lingodb/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/src/lingodb/utility/logging_tools.cpp
)
target_link_libraries(test_dsa_char_append
    gtest gtest_main
    MLIRDSA
    MLIRUtilDialect
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTransforms
)
add_test(NAME DSACharAppendTest COMMAND test_dsa_char_append)