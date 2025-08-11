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

# Enable position-independent code globally for shared library compatibility
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set default build type to Debug only if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
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

include(PostgreSQLExtension)

# Include directories for the new structure
include_directories(
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_BINARY_DIR}/include/pgx_lower
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

# Add include directory for TableGen processing
add_subdirectory(include/pgx_lower/mlir/Dialect/RelAlg/IR)
add_subdirectory(include/pgx_lower/mlir/Dialect/DB/IR)
add_subdirectory(include/pgx_lower/mlir/Dialect/DSA/IR)
# Add include directory for util dialect
add_subdirectory(include/pgx_lower/mlir/Dialect/util)

# Add compiler library (includes dialects)
add_subdirectory(src/compiler)

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
    # Clean slate refactor - minimal unit tests for core infrastructure only
    file(GLOB_RECURSE UNIT_TEST_SOURCES 
        "tests/unit/core/*.cpp"
        "tests/unit/runtime/*.cpp"
        "tests/unit/logging/*.cpp"
        "tests/unit/dialects/*.cpp"
        "tests/unit/dialect_dsa/*.cpp"  # Added for new DSA operations tests
        "tests/unit/frontend/*.cpp"
        "tests/unit/conversion/*.cpp"
        "tests/unit/mlir/*.cpp"  # Added for pass pipeline tests
        "tests/unit/execution/*.cpp"  # Added for JIT execution engine tests
    )
    # mlir_runner_test.cpp now included - minimal MLIR context and dialect loading implemented
    
    # NOTE: test_expression_handling.cpp and test_runtime_wrapper_pattern.cpp
    # temporarily removed due to clean slate refactor - they reference deleted MLIR components
    # Will be restored once basic dialects are rebuilt
    
    # Build list of additional sources based on configuration
    set(UNIT_TEST_ADDITIONAL_SOURCES
        src/frontend/SQL/query_analyzer.cpp
        src/frontend/SQL/postgresql_ast_translator.cpp
        src/execution/error_handling.cpp
        src/execution/mlir_runner.cpp
        src/execution/jit_execution_engine.cpp  # Phase 4g-2a JIT infrastructure
        src/execution/jit_execution_wrapper.cpp  # Phase 4g-2c header isolation
        src/utility/logging.cpp
        src/utility/logging_tools.cpp
        src/utility/ir_debug_utils.cpp
        src/runtime/helpers.cpp
        # Note: PostgreSQL runtime files require full PostgreSQL runtime
        # and cannot be easily compiled in unit test environment
            # Added for centralized pass pipeline
    )
    
    
    add_mlir_unit_test(mlir_unit_test 
        tests/unit/test_main.cpp
        tests/unit/test_helpers.cpp
        tests/unit/mocks/mlir_runner_mock.cpp
        tests/unit/mocks/runtime_mocks.cpp  # Mock runtime functions for symbol registration
        ${UNIT_TEST_SOURCES}
        ${UNIT_TEST_ADDITIONAL_SOURCES}
    )
    target_link_libraries(mlir_unit_test PRIVATE 
        gtest
        MLIRRelAlgDialect
        MLIRDBDialect
        MLIRDSADialect
        MLIRUtilDialect               # Util dialect for memory management and tuples
        MLIRRelAlgToDSAConversion
        MLIRRelAlgToDBConversion      # New RelAlg -> DB conversion
        MLIRDBToStdConversion         # New DB -> Standard conversion (Phase 4d)
        MLIRDSAToStdConversion        # DSA -> Standard conversion (Phase 4e-2)
        MLIRDSAToLLVMConversion       # DSA → LLVM conversion for data structure algorithms
        UtilToLLVM                    # Util → LLVM conversion (Phase 4f)
        MLIRPgxLowerPasses            # Centralized pass pipeline infrastructure
        MLIRFuncDialect
        MLIRArithDialect
        MLIRSCFDialect
        MLIRMemRefDialect             # Needed for memref operations
        MLIRIndexDialect              # Needed for index operations
        MLIRLLVMDialect               # Needed for DSA -> LLVM conversion
        MLIRLLVMCommonConversion      # LLVM conversion infrastructure
        MLIRFuncToLLVM                # Function to LLVM conversion patterns
        MLIRExecutionEngine           # JIT execution engine for Phase 4g-2a
        MLIRTargetLLVMIRExport        # LLVM IR export functionality
        MLIRPass
        MLIRTransforms
        MLIRParser
        # NOTE: Additional MLIR dialect libraries will be added as they are rebuilt
    )
    target_include_directories(mlir_unit_test PRIVATE 
        tests/unit
    )
    target_compile_definitions(mlir_unit_test PRIVATE BUILDING_UNIT_TESTS)
endif()
