# Runtime Unit Tests

# Test SPI Functions
add_executable(test_spi_functions test_spi_functions.cpp)
target_link_libraries(test_spi_functions
    gtest gtest_main
    pgx_lower_runtime
    pgx_lower_common
)
add_test(NAME SPIFunctionsTest COMMAND test_spi_functions)

# Additional runtime tests can be added here