# Runtime Unit Tests

# Test SPI Functions
add_executable(test_spi_functions test_spi_functions.cpp)
target_link_libraries(test_spi_functions
    gtest gtest_main
    pgx_lower_runtime
    pgx_lower_common
)
add_test(NAME SPIFunctionsTest COMMAND test_spi_functions)

# Test Memory Context Safety
add_executable(test_memory_context_safety test_memory_context_safety.cpp)
target_link_libraries(test_memory_context_safety
    gtest gtest_main
    pgx_lower_runtime
    pgx_lower_common
)
add_test(NAME MemoryContextSafetyTest COMMAND test_memory_context_safety)

# Test PostgreSQL SPI Integration
add_executable(test_postgresql_spi test_postgresql_spi.cpp)
target_link_libraries(test_postgresql_spi
    gtest gtest_main
    pgx_lower_runtime
    pgx_lower_common
)
add_test(NAME PostgreSQLSPITest COMMAND test_postgresql_spi)

# Additional runtime tests can be added here