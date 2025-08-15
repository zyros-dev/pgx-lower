# PostgreSQL Library Tests (Category B)
# These tests link PostgreSQL libraries but assume DB is offline
# Target: Memory context crashes, exception handling, API linking

# Required for PostgreSQL memory context and exception handling
find_package(PostgreSQL REQUIRED)

# Create PostgreSQL library test executable
add_executable(pgx_lower_postgres_tests
    test_postgresql_memory_context.cpp
    test_postgresql_exception_handling.cpp
    test_runtime_function_linking.cpp
)

# Link PostgreSQL libraries for memory management and exception handling
target_link_libraries(pgx_lower_postgres_tests
    PRIVATE
    gtest
    gtest_main
    # PostgreSQL core libraries (no connection required)
    ${POSTGRESQL_LIBRARIES}
    # MLIR and LLVM for JIT testing
    ${LLVM_LIBRARIES}
    # Our core libraries
    pgx_lower_core
    pgx_lower_jit
    pgx_lower_logging
)

# Include PostgreSQL headers for memory context API
target_include_directories(pgx_lower_postgres_tests
    PRIVATE
    ${POSTGRESQL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${LLVM_INCLUDE_DIRS}
)

# Compiler definitions for PostgreSQL integration
target_compile_definitions(pgx_lower_postgres_tests
    PRIVATE
    # PostgreSQL version and API compatibility
    PG_VERSION_NUM=${POSTGRESQL_VERSION_NUM}
    # Enable PostgreSQL memory context debugging
    MEMORY_CONTEXT_CHECKING=1
    # Enable exception stack trace
    USE_ASSERT_CHECKING=1
)

# Special compiler flags for PostgreSQL integration
target_compile_options(pgx_lower_postgres_tests
    PRIVATE
    # PostgreSQL requires specific flags
    -fPIC
    -D_GNU_SOURCE
    # Enable debugging symbols for crash analysis
    -g
    -O0
)

# Register tests with CTest
add_test(
    NAME pgx_lower_postgres_tests
    COMMAND pgx_lower_postgres_tests
)

# Set test environment for PostgreSQL library tests
set_tests_properties(pgx_lower_postgres_tests
    PROPERTIES
    ENVIRONMENT "PGXLOWER_LOG_LEVEL=DEBUG"
)