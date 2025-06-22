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

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(pgx_lower LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Note: MLIR headers may produce C++20 ambiguity warnings, but they're not critical

include(PostgreSQLConfig)
find_package(PostgreSQL REQUIRED)
find_package(LLVM REQUIRED)
find_package(MLIR REQUIRED)

include(PostgreSQLExtension)

# Include directories for the new structure
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${PostgreSQL_INCLUDE_DIRS}
    ${PostgreSQL_SERVER_INCLUDE_DIRS}
)

enable_testing()

# Add dialect library
add_subdirectory(src/dialects)

# Add the main extension
add_subdirectory(extension)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(NOT DEFINED BUILD_ONLY_EXTENSION OR NOT BUILD_ONLY_EXTENSION)
    add_mlir_unit_test(mlir_unit_test 
        tests/unit/mlir_unit_test.cpp
        src/core/mlir_runner.cpp
        src/core/mlir_code_generator.cpp
        src/core/query_analyzer.cpp
        src/core/error_handling.cpp
        src/logging/mlir_logger.cpp
        src/runtime/tuple_access.cpp
    )
    target_link_libraries(mlir_unit_test PRIVATE 
        gtest 
        gtest_main
    )
endif()
