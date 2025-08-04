# PostgreSQL Extension Configuration for pgx-lower
# Clean, organized extension build using the new directory structure

add_postgresql_mixed_extension(
        pgx_lower
        VERSION 1.0
        C_SOURCES 
            ../src/execution/postgres/executor_c.c
        CPP_SOURCES
            ../src/execution/postgres/my_executor.cpp 
            ../src/execution/postgres/executor_c.cpp
            ../src/execution/mlir_runner.cpp 
            ../src/execution/postgresql_ast_translator.cpp
            ../src/execution/query_analyzer.cpp
            ../src/execution/error_handling.cpp
            ../src/utility/logging.cpp
            ../src/utility/logging_c.cpp
            ../src/utility/logging_tools.cpp
            ../src/utility/logging/mlir_logger_pg.cpp
            ../src/utility/logging/console_logger.cpp
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
            14_aggregate_functions
            15_special_operators
)

# Link the MLIR dialect libraries
# Note: MLIRPgDialect removed as we deleted the PG dialect
target_link_libraries(pgx_lower PRIVATE 
    -Wl,--no-as-needed
    -Wl,--whole-archive
    PgxLowerRuntime  # Re-added: needed for RuntimeCallGenerator implementation
    -Wl,--no-whole-archive
    MLIRSubOpDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRTupleStreamDialect
    MLIRRelAlgDialect
    MLIRUtilDialect
    # Add conversion libraries for MLIR lowering passes
    MLIRRelAlgToSubOp
    MLIRSubOpToControlFlow  # Fixes: createLowerSubOpPass() missing symbol
    MLIRDBToDSA
    MLIRDBToLLVM
    MLIRDSAToLLVM
    MLIRUtilToLLVM
    -Wl,--as-needed
)

# Add include directories for source files
target_include_directories(pgx_lower PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

