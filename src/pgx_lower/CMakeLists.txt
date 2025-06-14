# Copyright 2020 Mats Kindahl
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
add_postgresql_mixed_extension(
        pgx_lower
        VERSION 1.0
        C_SOURCES src/executor_c.c
        CPP_SOURCES src/my_executor.cpp
        SCRIPTS pgx_lower--1.0.sql
        REGRESS basic)

#add_library(pgx_lower MODULE src/executor_c.c src/executor.cpp)
#
#target_link_libraries(pgx_lower PRIVATE
#        MLIRIR
#        MLIRParser
#        MLIRSupport
#        MLIRTransforms
#        LLVMCore
#        LLVMSupport
#)
#
#set_target_properties(pgx_lower PROPERTIES
#        PREFIX ""
#        POSITION_INDEPENDENT_CODE ON
#        BUILD_WITH_INSTALL_RPATH TRUE
#        INSTALL_RPATH "${ARROW_LIB_PATH}"
#)
#
#target_include_directories(pgx_lower
#        PRIVATE ${PostgreSQL_SERVER_INCLUDE_DIRS}
#        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#)
#
## Generate control file
#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/pgx_lower.control.in
#        ${CMAKE_CURRENT_BINARY_DIR}/pgx_lower.control
#        @ONLY
#)
#
## Install extension files
#install(TARGETS pgx_lower
#        LIBRARY DESTINATION ${PostgreSQL_PACKAGE_LIBRARY_DIR}
#)
#
#install(FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/pgx_lower.control
#        ${CMAKE_CURRENT_SOURCE_DIR}/pgx_lower--1.0.sql
#        DESTINATION ${PostgreSQL_EXTENSION_DIR}
#)
#
## Add regression tests for all .sql files in sql/
#file(GLOB PGX_SQL_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../sql ${CMAKE_CURRENT_SOURCE_DIR}/../sql/*.sql)
#foreach(sql_file ${PGX_SQL_TESTS})
#    string(REPLACE ".sql" "" test_name ${sql_file})
#    if(PG_REGRESS)
#        add_test(
#            NAME pgx_lower_${test_name}
#            COMMAND
#            ${PG_REGRESS} --temp-instance=${CMAKE_BINARY_DIR}/tmp_check
#            --inputdir=${CMAKE_CURRENT_SOURCE_DIR}/../sql
#            --outputdir=${CMAKE_CURRENT_BINARY_DIR}/regress_out
#            --expecteddir=${CMAKE_CURRENT_SOURCE_DIR}/../expected
#            --load-extension=pgx_lower
#            ${test_name}
#        )
#    endif()
#endforeach()
#
#link_directories(/usr/lib/postgresql/17/lib)
