# CMake generated Testfile for 
# Source directory: /home/xzel/repos/pgx-lower/extension
# Build directory: /home/xzel/repos/pgx-lower/extension
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[pgx_lower_regress]=] "/usr/local/pgsql/lib/pgxs/src/test/regress/pg_regress" "--bindir=/usr/local/pgsql/bin" "--dlpath=/usr/local/pgsql/lib" "--inputdir=/home/xzel/repos/pgx-lower/tests" "--outputdir=/home/xzel/repos/pgx-lower/extension" "--load-extension=pgx_lower" "1_one_tuple" "2_two_tuples" "3_lots_of_tuples" "4_two_columns_ints" "5_two_columns_diff" "6_every_type" "7_sub_select" "8_subset_all_types" "9_basic_arithmetic_ops" "10_comparison_ops" "11_logical_ops" "12_null_handling" "13_text_operations" "14_aggregate_functions" "15_special_operators")
set_tests_properties([=[pgx_lower_regress]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/xzel/repos/pgx-lower/cmake/PostgreSQLExtension.cmake;127;add_test;/home/xzel/repos/pgx-lower/extension/CMakeLists.txt;4;add_postgresql_mixed_extension;/home/xzel/repos/pgx-lower/extension/CMakeLists.txt;0;")
