# PostgreSQL Extension Configuration for pgx-lower
if(NOT BUILDING_UNIT_TESTS)
    add_postgresql_mixed_extension(
        pgx_lower
        VERSION 1.0
        C_SOURCES
        ../src/pgx-lower/execution/postgres/executor_c.c
        CPP_SOURCES
        ../src/pgx-lower/execution/postgres/my_executor.cpp
        ../src/pgx-lower/execution/postgres/executor_c.cpp
        ../src/pgx-lower/execution/mlir_runner.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_utils.cpp
        ../src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp
        ../src/pgx-lower/execution/jit_engine/jit_execution_wrapper.cpp
        ../src/pgx-lower/frontend/SQL/postgresql_ast_translator.cpp
        ../src/pgx-lower/frontend/SQL/translation/expression_translator.cpp
        ../src/pgx-lower/frontend/SQL/translation/plan_translator.cpp
        ../src/pgx-lower/frontend/SQL/translation/schema_manager.cpp
        ../src/pgx-lower/frontend/SQL/translation/translation_core.cpp
        ../src/pgx-lower/frontend/SQL/query_analyzer.cpp
        ../src/pgx-lower/utility/error_handling.cpp
        ../src/pgx-lower/utility/logging.cpp
        ../src/pgx-lower/utility/logging_c.cpp
        ../src/pgx-lower/runtime/tuple_access.cpp
        ../src/pgx-lower/runtime/PostgreSQLDataSource.cpp
        ../src/pgx-lower/runtime/PostgreSQLRuntime.cpp
        ../src/pgx-lower/runtime/DateRuntime.cpp
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
            26_before_check_types
            27_group_by_having
            28_group_by_with_where
            29_expressions_in_aggregations
            30_test_missing_expressions
            31_distinct_statement
            32_decimal_maths
            33_basic_joins
            34_advanced_joins
            35_nested_queries
            36_tpch_minimal
            37_tpch_minimal_2
            38_tpch_minimal_3
            39_tpch_minimal
            init_tpch
            40_tpch_not_lowered
            tpch
            tpch_no_lower
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
    runtime
    -Wl,--no-whole-archive
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRFuncToLLVM
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    -Wl,--as-needed
)

target_link_options(pgx_lower PRIVATE -Wl,--export-dynamic)

add_dependencies(pgx_lower
    PGXLowerRelAlgOpsIncGen
    PGXLowerDBOpsIncGen
    PGXLowerDSAOpsIncGen
    MLIRUtilOpsIncGen
)

target_include_directories(pgx_lower PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

endif() # NOT BUILDING_UNIT_TESTS

