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
        ../src/execution/mlir_setup/mlir_runner_phases.cpp
        ../src/execution/jit_engine/jit_execution_engine.cpp
        ../src/execution/jit_engine/jit_execution_wrapper.cpp
            ../src/frontend/SQL/postgresql_ast_translator.cpp
            ../src/frontend/SQL/query_analyzer.cpp
        ../src/utility/error_handling.cpp
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
    MLIRDSA
    MLIRUtilDialect
    MLIRRelAlgToDB
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRUtilToLLVM
    StandardToLLVM
    MLIRPasses
    MLIRCustomTransforms
    mlir-support
    -Wl,--no-whole-archive
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRFuncToLLVM
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    -Wl,--as-needed
)

target_link_options(pgx_lower PRIVATE -Wl,--export-dynamic)

# Add dependencies on TableGen targets to ensure .inc files are generated first
add_dependencies(pgx_lower 
    PGXLowerRelAlgOpsIncGen
    PGXLowerDBOpsIncGen
    PGXLowerDSAOpsIncGen
    MLIRUtilOpsIncGen
)

target_include_directories(pgx_lower PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

