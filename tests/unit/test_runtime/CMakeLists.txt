find_package(GTest REQUIRED)

add_executable(test_numeric_conversion
    test_numeric_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/runtime/NumericConversion.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_lowerings/pg_stubs.cpp
)

target_compile_definitions(test_numeric_conversion PRIVATE POSTGRESQL_EXTENSION=1)

target_include_directories(test_numeric_conversion PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${PostgreSQL_INCLUDE_DIRS}
    ${PostgreSQL_SERVER_INCLUDE_DIRS}
)

target_link_libraries(test_numeric_conversion GTest::gtest_main)

add_test(NAME NumericConversionTest COMMAND test_numeric_conversion)
