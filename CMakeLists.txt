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

# Coverage support
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang coverage
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate")
    else()
        # GCC coverage
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()
endif()

# Note: MLIR headers may produce C++20 ambiguity warnings, but they're not critical
# Disable specific warnings that cause build failures with LLVM 20 and C++20
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang specific warning suppression
    add_compile_options(-Wno-ambiguous-reversed-operator)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC specific warning suppression - completely disable warning-as-error
    add_compile_options(-Wno-error)
    add_compile_options(-w)  # Suppress all warnings temporarily to bypass MLIR C++20 issues
    # Also suppress specific MLIR-related warnings for GCC
    add_compile_options(-Wno-unused-parameter -Wno-unused-variable)
endif()

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

# Create build_includes target for runtime header generation
add_custom_target(build_includes)
set(runtime_funcs_ptr_libs "")

# Add tools for runtime header generation
add_subdirectory(tools/build-tools)

# Skip runtime header generation for now - we'll use manual wrappers
# TODO: Fix runtime-header-tool to work with system clang libraries

# Create empty runtime_funcs_ptr library for now
add_library(runtime_funcs_ptr INTERFACE)

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
    # Collect all test source files (excluding mlir_runner_test.cpp which needs full MLIR pipeline)
    file(GLOB_RECURSE UNIT_TEST_SOURCES 
        "tests/unit/core/*.cpp"
        "tests/unit/dialects/*.cpp"
        "tests/unit/runtime/*.cpp"
    )
    list(REMOVE_ITEM UNIT_TEST_SOURCES 
        "${CMAKE_SOURCE_DIR}/tests/unit/core/mlir_runner_test.cpp"
    )
    
    # Add individual test files
    list(APPEND UNIT_TEST_SOURCES 
        tests/unit/test_subop_simple.cpp
    )
    
    add_mlir_unit_test(mlir_unit_test 
        tests/unit/test_main.cpp
        tests/unit/test_helpers.cpp
        tests/unit/mocks/mlir_runner_mock.cpp
        ${UNIT_TEST_SOURCES}
        src/core/query_analyzer.cpp
        src/core/error_handling.cpp
        src/core/logging.cpp
        src/logging/console_logger.cpp
        src/runtime/helpers.cpp
    )
    target_link_libraries(mlir_unit_test PRIVATE 
        gtest
        MLIRRelAlgDialect
        MLIRSubOpDialect
        MLIRDBDialect
        MLIRDSADialect
        MLIRTupleStreamDialect
        MLIRUtilDialect
    )
    target_include_directories(mlir_unit_test PRIVATE 
        tests/unit
    )
    target_compile_definitions(mlir_unit_test PRIVATE BUILDING_UNIT_TESTS)
endif()
