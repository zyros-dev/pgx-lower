# RelAlg Unit Tests CMakeLists.txt

add_executable(test_relalg_basic test_relalg_basic.cpp)

target_link_libraries(test_relalg_basic
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRIR
    MLIRArithmeticDialect
)

target_include_directories(test_relalg_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests/unit
)

add_test(NAME RelAlgBasicTest COMMAND test_relalg_basic)