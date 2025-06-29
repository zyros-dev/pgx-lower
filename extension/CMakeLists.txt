# PostgreSQL Extension Configuration for pgx-lower
# Clean, organized extension build using the new directory structure

add_postgresql_mixed_extension(
        pgx_lower
        VERSION 1.0
        C_SOURCES 
            ../src/postgres/executor_c.c
        CPP_SOURCES
            ../src/postgres/my_executor.cpp 
            ../src/postgres/executor_c.cpp
            ../src/core/mlir_runner.cpp 
            ../src/core/mlir_builder.cpp 
            ../src/core/query_analyzer.cpp
            ../src/core/error_handling.cpp
            ../src/core/logging.cpp
            ../src/logging/mlir_logger_pg.cpp
        SCRIPTS 
            sql/pgx_lower--1.0.sql
        REGRESS
            1_one_tuple
            2_two_tuples
            3_lots_of_tuples
            4_two_columns_ints
            5_two_columns_diff
            6_every_type
            7_sub_select
            8_subset_all_types
            9_basic_arithmetic_ops
            10_comparison_ops
            11_logical_ops
            12_null_handling
            13_text_operations
)

# Link the PostgreSQL dialect and runtime libraries
target_link_libraries(pgx_lower PRIVATE 
    MLIRPgDialect
    PgxLowerRuntime
)

# Add include directories for source files
target_include_directories(pgx_lower PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

