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
            ../src/execution/unit_test_from_postgres.cpp 
            ../src/execution/jit_execution_engine.cpp
            ../src/execution/jit_execution_wrapper.cpp
            ../src/frontend/SQL/postgresql_ast_translator.cpp
            ../src/frontend/SQL/query_analyzer.cpp
            ../src/execution/error_handling.cpp
            ../src/utility/logging.cpp
            ../src/utility/logging_c.cpp
            ../src/utility/logging_tools.cpp
            ../src/runtime/tuple_access.cpp
            ../src/runtime/PostgreSQLDataSource.cpp
            ../src/runtime/helpers.cpp
            ../src/runtime/postgresql_runtime.cpp
            ../src/runtime/MetaData.cpp
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
            16_debug_text
            17_where_simple_conditions
            18_where_logical_combinations
            19_where_null_patterns
            20_where_pattern_matching
            21_order_by_basic
            22_order_by_multiple_columns
            23_order_by_expressions
            24_order_by_with_where
            25_group_by_simple
            26_group_by_having
            27_group_by_with_where
            28_complete_sql_patterns
)

# MLIR dialect libraries for TypeID resolution  
# These are required for extension loading to resolve MLIR dialect TypeIDs
target_link_libraries(pgx_lower PRIVATE 
    -Wl,--no-as-needed
    -Wl,--whole-archive
    MLIRRelAlgDialect
    MLIRDBDialect  
    MLIRDSA                        # Our custom DSA dialect
    MLIRUtilDialect                # Util dialect for memory management and tuples
    MLIRRelAlgToDB
    MLIRDBToArrowStd               # DB → Standard MLIR  
    MLIRDSAToStd                   # DSA → Standard MLIR
    MLIRUtilToLLVM                 # Util → LLVM conversion
    StandardToLLVM                 # Unified Standard→LLVM conversion pass
    MLIRPasses                     # Central pass registration
    # PgxLowerRuntime  # TODO: Re-add when runtime system is rebuilt
    -Wl,--no-whole-archive
    # Standard MLIR dialects required by our dialects
    MLIRLLVMDialect  # Required for DSA → LLVM conversion
    MLIRLLVMCommonConversion  # LLVM conversion infrastructure
    MLIRFuncToLLVM  # Function to LLVM conversion patterns
    MLIRExecutionEngine  # JIT execution engine for Phase 4g-2a
    MLIRTargetLLVMIRExport  # LLVM IR export functionality
    -Wl,--as-needed
)

# Add include directories for source files
target_include_directories(pgx_lower PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

