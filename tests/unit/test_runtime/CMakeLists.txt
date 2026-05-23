# Unit tests for src/pgx-lower/runtime/*.cpp.
#
# Spec 16: these tests link against real PG headers and a curated subset of
# PG backend symbols. The runtime FFI code is too tightly coupled to PG's
# Datum / Numeric / varlena machinery for stub-based testing to be honest.
#
# Strategy (iterative): start with the existing pg_stubs.cpp from
# test_lowerings/, add real PG .o files only when stubs can't honestly
# satisfy a path. test_numeric_conversion is the first proof point: the
# forward direction (numeric_to_i128) only needs pg_detoast_datum + the
# numeric structure macros (header-only), so it links cleanly with the
# existing stubs.

find_package(GTest REQUIRED)
find_package(PostgreSQL REQUIRED)

add_executable(test_numeric_conversion
    test_numeric_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/runtime/NumericConversion.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/pgx-lower/utility/logging_c.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_lowerings/pg_stubs.cpp
)

target_compile_definitions(test_numeric_conversion PRIVATE
    BUILDING_UNIT_TESTS=1
)

target_include_directories(test_numeric_conversion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lingo-db/include
    ${CMAKE_SOURCE_DIR}/src
    ${PostgreSQL_INCLUDE_DIRS}
    ${PostgreSQL_SERVER_INCLUDE_DIRS}
)

target_link_libraries(test_numeric_conversion
    GTest::gtest_main
)

add_test(NAME NumericConversionTest COMMAND test_numeric_conversion)
set_tests_properties(NumericConversionTest PROPERTIES LABELS "unit;runtime;numeric")
