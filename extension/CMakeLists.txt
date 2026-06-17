# PostgreSQL Extension Configuration for pgx-lower
if(NOT BUILDING_UNIT_TESTS)
    set(_pgx_lower_test_sources)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        list(APPEND _pgx_lower_test_sources
            ../src/pgx-lower/test/numeric_tests.cpp
            ../src/pgx-lower/test/type_mapping_tests.cpp
            ../src/pgx-lower/test/boolean_lowering_tests.cpp
            ../src/pgx-lower/test/ast_const_translation_tests.cpp
            ../src/pgx-lower/test/expression_metadata_translation_tests.cpp
            ../src/pgx-lower/test/standalone_mlir_runner.cpp
            ../src/pgx-lower/test/pipeline_phases_tests.cpp
            ../src/pgx-lower/test/query_analyzer_tests.cpp
            ../src/pgx-lower/test/accepted_plan_verifier_tests.cpp
            ../src/pgx-lower/test/pg_primitive_type_tests.cpp
            ../src/pgx-lower/test/pg_sql_expression_nullability_tests.cpp
            ../src/pgx-lower/test/pg_bool_predicate_truth_tests.cpp
            ../src/pgx-lower/test/pg_physicalization_tests.cpp
            ../src/pgx-lower/test/pg_relalg_nullability_tests.cpp
            ../src/pgx-lower/test/pg_runtime_function_type_tests.cpp
            ../src/pgx-lower/test/pg_nullability_tests.cpp
            ../src/pgx-lower/test/pg_legacy_nullable_cleanup_tests.cpp
            ../src/pgx-lower/test/postgresql_type_mapper_tests.cpp
        )
    endif()

    set(PGX_CLI_EXECUTABLE "${CMAKE_SOURCE_DIR}/pgx-cli/dist/index.js")
    if(NOT EXISTS "${PGX_CLI_EXECUTABLE}")
        message(FATAL_ERROR "Missing in-repo pgx-cli/dist/index.js; build pgx-cli before configuring CMake")
    endif()

    add_postgresql_mixed_extension(
        pgx_lower
        VERSION 1.0
        C_SOURCES
        ../src/pgx-lower/execution/postgres/executor_c.c
        CPP_SOURCES
        ${_pgx_lower_test_sources}
        ../src/pgx-lower/execution/postgres/my_executor.cpp
        ../src/pgx-lower/execution/postgres/executor_c.cpp
        ../src/pgx-lower/execution/accepted_plan_verifier.cpp
        ../src/pgx-lower/execution/mlir_runner.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp
        ../src/pgx-lower/execution/mlir_setup/mlir_runner_utils.cpp
        ../src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp
        ../src/pgx-lower/frontend/SQL/postgresql_ast_translator.cpp
        ../src/pgx-lower/frontend/SQL/translation/expression_translator_basic.cpp
        ../src/pgx-lower/frontend/SQL/translation/expression_translator_complex.cpp
        ../src/pgx-lower/frontend/SQL/translation/expression_translator_functions.cpp
        ../src/pgx-lower/frontend/SQL/translation/expression_translator_operators.cpp
        ../src/pgx-lower/frontend/SQL/translation/plan_translator_agg.cpp
        ../src/pgx-lower/frontend/SQL/translation/plan_translator_joins.cpp
        ../src/pgx-lower/frontend/SQL/translation/plan_translator_scans.cpp
        ../src/pgx-lower/frontend/SQL/translation/plan_translator_utils.cpp
        ../src/pgx-lower/frontend/SQL/translation/schema_manager.cpp
        ../src/pgx-lower/frontend/SQL/translation/translation_core.cpp
        ../src/pgx-lower/frontend/SQL/query_analyzer.cpp
        ../src/pgx-lower/utility/error_handling.cpp
        ../src/pgx-lower/utility/logging.cpp
        ../src/pgx-lower/utility/logging_c.cpp
        ../src/pgx-lower/utility/mlir_logging.cpp
        ../src/pgx-lower/runtime/tuple_access.cpp
        ../src/pgx-lower/runtime/PostgreSQLDataSource.cpp
        ../src/pgx-lower/runtime/PostgreSQLRuntime.cpp
        ../src/pgx-lower/runtime/NumericRuntime.cpp
        ../src/pgx-lower/runtime/DateRuntime.cpp
        SCRIPTS
            sql/pgx_lower--1.0.sql
        REGRESS_INPUTDIR
            ${CMAKE_SOURCE_DIR}/tests/pgx-regression
        REGRESS_OUTPUTDIR
            ${CMAKE_CURRENT_BINARY_DIR}/pgx-regression
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
            40_tpch_not_lowered
            41_sorts
            42_test_relalg_function
            43_decode_at_scan
            44_numeric_correctness
            45_route_notices
            46_sql_3vl_expressions
            47_pg_bool_predicate_truth
            48_grouping_sets_fallback
            49_aggref_filter_fallback
            50_min_text_fallback
            51_aggorder_fallback
            52_aggdistinct_fallback
)


if(PG_REGRESS)
    add_test(
        NAME pgx_lower_regress_routes
        COMMAND ${CMAKE_COMMAND}
                -E
                env
                ${PGX_CLI_EXECUTABLE}
                test
                route-check
                --run-name pgx-regression-correctness
                --profile debug
                --execution-mode extension-auto
                --sql-dir ${CMAKE_SOURCE_DIR}/tests/pgx-regression/sql
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/pgx-regression/results
                --summary ${CMAKE_SOURCE_DIR}/build-artifacts/test-runs/pgx-regression-correctness/summary.md
                --default-auto-should-route-to lower
                --require-route-directives
    )
    set_tests_properties(pgx_lower_regress_routes PROPERTIES DEPENDS pgx_lower_regress)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/unit-tests/sql)
    file(CHMOD ${CMAKE_SOURCE_DIR}/tests/unit-tests/sql PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
    add_test(
        NAME pgx_lower_regress_unit
        COMMAND bash
                -lc
                "set -euo pipefail; cd '${CMAKE_SOURCE_DIR}'; rm -f '${CMAKE_SOURCE_DIR}'/tests/unit-tests/sql/*.sql; '${PGX_CLI_EXECUTABLE}' test unit-sql --root '${CMAKE_SOURCE_DIR}'; /usr/local/pgsql/bin/dropdb --if-exists regression_unit; /usr/local/pgsql/bin/createdb regression_unit; fail=0; for sql in '${CMAKE_SOURCE_DIR}'/tests/unit-tests/sql/*.sql; do echo \"--- $(basename \"$sql\") ---\"; /usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f \"$sql\" || { fail=1; echo FAIL: \"$sql\"; }; done; echo; if [ $fail -eq 0 ]; then echo UTEST-PG_OK; else echo UTEST-PG_FAILED; exit 1; fi"
    )
endif()

if(PG_REGRESS)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tpch)
    add_test(
        NAME pgx_lower_tpch_regress
        COMMAND ${PG_REGRESS}
                --bindir=${_pg_bindir}
                --dlpath=${PostgreSQL_PACKAGE_LIBRARY_DIR}
                --inputdir=${CMAKE_SOURCE_DIR}/tests/tpch
                --outputdir=${CMAKE_CURRENT_BINARY_DIR}/tpch
                --load-extension=pgx_lower
                init_tpch
                tpch_no_lower
                tpch
    )

    add_test(
        NAME pgx_lower_tpch_routes
        COMMAND ${CMAKE_COMMAND}
                -E
                env
                ${PGX_CLI_EXECUTABLE}
                test
                route-check
                --run-name tpch-correctness
                --profile debug
                --execution-mode extension-auto
                --sql-dir ${CMAKE_SOURCE_DIR}/tests/tpch/sql
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/tpch/results
                --summary ${CMAKE_SOURCE_DIR}/build-artifacts/test-runs/tpch-correctness/summary.md
                --default-auto-should-route-to lower
                --require-route-directives
    )
    set_tests_properties(pgx_lower_tpch_routes PROPERTIES DEPENDS pgx_lower_tpch_regress)

    add_test(
        NAME pgx_lower_compare_postgres_tpch
        COMMAND ${CMAKE_COMMAND}
                -E
                env
                PGX_COMPARE_POSTGRES_INTERNAL=1
                ${PGX_CLI_EXECUTABLE}
                test
                compare-postgres-internal
                --workload tpch-correctness
                --root ${CMAKE_SOURCE_DIR}
                --output-dir ${CMAKE_SOURCE_DIR}/build-artifacts/test-runs/ctest-tpch-correctness/compare-postgres
                --from-managed-runner
    )
    set_tests_properties(pgx_lower_compare_postgres_tpch PROPERTIES DEPENDS pgx_lower_tpch_regress)
endif()

if(PGX_LOWER_ENABLE_POSTGRES_REGRESS)
    if(NOT PG_REGRESS)
        message(FATAL_ERROR "PGX_LOWER_ENABLE_POSTGRES_REGRESS requires pg_regress")
    endif()
    add_test(
        NAME pgx_lower_psql_regression_burndown
        COMMAND ${CMAKE_COMMAND}
                -E
                env
                ${PGX_CLI_EXECUTABLE}
                test
                psql-regression-burndown
                --source ${CMAKE_SOURCE_DIR}/tests/psql-regression
                --baseline ${CMAKE_SOURCE_DIR}/tests/psql-regression/baselines/current.txt
                --output-dir ${CMAKE_SOURCE_DIR}/tests/psql-regression/results
                --summary ${CMAKE_SOURCE_DIR}/build-artifacts/test-runs/psql-regression-burndown/summary.md
                --route-summary ${CMAKE_SOURCE_DIR}/build-artifacts/test-runs/psql-regression-burndown/route-summary.md
                --pg-regress ${PG_REGRESS}
                --bindir ${_pg_bindir}
                --dlpath ${PostgreSQL_PACKAGE_LIBRARY_DIR}
    )
endif()

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

target_precompile_headers(pgx_lower PRIVATE
    ../src/pgx-lower/frontend/SQL/translation/translation_pch.h
)

endif() # NOT BUILDING_UNIT_TESTS
