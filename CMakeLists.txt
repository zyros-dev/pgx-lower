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

# Enforce Clang compiler due to GCC 14.2.0 -O3 bug (infinite loop in mlir::reconcileUnrealizedCasts)
# Set compilers before project() call if not already set
if(NOT DEFINED ENV{CC})
    set(ENV{CC} "clang")
endif()
if(NOT DEFINED ENV{CXX})
    set(ENV{CXX} "clang++")
endif()

project(pgx_lower LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# Verify we got Clang after project() detects compiler
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "This project requires Clang due to GCC optimizer bugs at -O3.\n"
        "Detected compiler: ${CMAKE_CXX_COMPILER_ID}\n"
        "Please clean your build directory and run: CC=clang CXX=clang++ make ptest")
endif()

# Enable position-independent code globally for shared library compatibility
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set default build type to Debug only if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2 -g")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g -O2")

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
# Set MLIR_DIR to the LLVM 20 installation
set(MLIR_DIR "/usr/lib/llvm-20/lib/cmake/mlir")
find_package(MLIR REQUIRED)


# Include MLIR CMake modules for TableGen
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# Add Clang CMake modules for runtime-header-tool
list(APPEND CMAKE_MODULE_PATH "/usr/lib/llvm-20/lib/cmake/clang")
include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(AddClang)
set("CLANG_VERSION" ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})

# Set MLIR TableGen include directories
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})

include(PostgreSQLExtension)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendored
    ${PostgreSQL_INCLUDE_DIRS}
    ${PostgreSQL_SERVER_INCLUDE_DIRS}
)

enable_testing()

# Enable compile commands export for IDE/linter integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global preprocessor definition for CLion linting support
# This ensures ifdef POSTGRESQL_EXTENSION blocks are properly linted
if(BUILDING_UNIT_TESTS)
    message(STATUS "Unit test mode - PostgreSQL extension disabled")
else()
    add_compile_definitions(POSTGRESQL_EXTENSION=1)
    message(STATUS "PostgreSQL extension mode enabled")
endif()

# Create build_includes target for runtime header generation
add_custom_target(build_includes)
set(runtime_funcs_ptr_libs "")

# Add tools for runtime header generation
add_subdirectory(tools/build-tools)


add_library(runtime_funcs_ptr INTERFACE)
add_subdirectory(include/lingodb/mlir/Dialect/RelAlg/IR)
add_subdirectory(include/lingodb/mlir/Dialect/DB/IR)
add_subdirectory(include/lingodb/mlir/Dialect/DSA/IR)
add_subdirectory(include/lingodb/mlir/Dialect/util)
add_subdirectory(src/lingodb/mlir)
add_subdirectory(src/lingodb/mlir-support)

add_subdirectory(src/lingodb/runtime)
add_subdirectory(src/lingodb/utility)

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
    # Add test subdirectories to build individual unit tests
    add_subdirectory(tests)
    # Add tools for development and testing
    add_subdirectory(tools)
    # Clean slate refactor - minimal unit tests for core infrastructure only
    file(GLOB_RECURSE UNIT_TEST_SOURCES 
        "tests/unit/core/*.cpp"
        "tests/unit/runtime/*.cpp"
        "tests/unit/logging/*.cpp"
        "tests/unit/dialects/*.cpp"
        "tests/unit/dialect_dsa/*.cpp"
        "tests/unit/frontend/*.cpp"
        "tests/unit/conversion/*.cpp"
        "tests/unit/mlir/*.cpp"
        "tests/unit/execution/*.cpp"
    )

    # Build list of additional sources based on configuration
    set(UNIT_TEST_ADDITIONAL_SOURCES
            src/pgx-lower/frontend/SQL/query_analyzer.cpp
            src/pgx-lower/frontend/SQL/postgresql_ast_translator.cpp
            src/pgx-lower/frontend/SQL/translation/expression_translator_basic.cpp
            src/pgx-lower/frontend/SQL/translation/expression_translator_complex.cpp
            src/pgx-lower/frontend/SQL/translation/expression_translator_functions.cpp
            src/pgx-lower/frontend/SQL/translation/expression_translator_operators.cpp
            src/pgx-lower/frontend/SQL/translation/plan_translator_agg.cpp
            src/pgx-lower/frontend/SQL/translation/plan_translator_joins.cpp
            src/pgx-lower/frontend/SQL/translation/plan_translator_scans.cpp
            src/pgx-lower/frontend/SQL/translation/plan_translator_utils.cpp
            src/pgx-lower/frontend/SQL/translation/schema_manager.cpp
            src/pgx-lower/frontend/SQL/translation/translation_core.cpp
            src/pgx-lower/utility/error_handling.cpp
            src/pgx-lower/execution/mlir_runner.cpp
            src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp
            src/pgx-lower/utility/logging.cpp
            src/lingodb/runtime/helpers.cpp
            src/lingodb/mlir/Passes.cpp
    )
    
    
endif()
