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

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

function(add_postgresql_mixed_extension NAME)
    set(_optional)
    set(_single VERSION)
    set(_multi C_SOURCES CPP_SOURCES SCRIPTS REGRESS)
    cmake_parse_arguments(_ext "${_optional}" "${_single}" "${_multi}" ${ARGN})

    if(NOT _ext_VERSION)
        message(FATAL_ERROR "Extension version not set")
    endif()

    add_library(${NAME} MODULE ${_ext_C_SOURCES} ${_ext_CPP_SOURCES})

    mlir_target_link_libraries(${NAME} PRIVATE
        LLVM 
        MLIR
    )
    target_link_libraries(${NAME} PRIVATE
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRControlFlowDialect
        MLIRSCFDialect
        MLIRFuncToLLVM
        MLIRMathToLLVM
        MLIRSCFToControlFlow
        MLIRControlFlowToLLVM
        MLIRArithToLLVM
        MLIRTransforms
    )
    set_target_properties(${NAME} PROPERTIES LINK_FLAGS "${_link_flags} -Wl,--no-as-needed")
    set_target_properties(${NAME} PROPERTIES INSTALL_RPATH "/usr/lib/llvm-20/lib")

    set(_link_flags "${PostgreSQL_SHARED_LINK_OPTIONS}")
    foreach(_dir ${PostgreSQL_SERVER_LIBRARY_DIRS})
        set(_link_flags "${_link_flags} -L${_dir}")
    endforeach()

    if(APPLE)
        set(_link_flags "${_link_flags} -bundle_loader ${PG_BINARY}")
    endif()

    set(FILTERED_PG_LIBRARIES "")
    foreach(lib ${PostgreSQL_LIBRARIES})
        if(NOT lib MATCHES "pgcommon|pgport")
            list(APPEND FILTERED_PG_LIBRARIES ${lib})
        endif()
    endforeach()
    
    target_link_libraries(${NAME} PRIVATE ${FILTERED_PG_LIBRARIES})

    set_target_properties(
        ${NAME}
        PROPERTIES
            PREFIX ""
            LINK_FLAGS "${_link_flags} -Wl,--no-as-needed"
            POSITION_INDEPENDENT_CODE ON
            INSTALL_RPATH "/usr/lib/llvm-20/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
    )

    target_include_directories(
        ${NAME}
        PRIVATE 
            ${PostgreSQL_SERVER_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    target_compile_definitions(${NAME} PRIVATE POSTGRESQL_EXTENSION)

    set(_control_file "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.control")
    file(
        GENERATE
        OUTPUT ${_control_file}
        CONTENT
            "# This file is generated content from add_postgresql_mixed_extension.
# No point in modifying it, it will be overwritten anyway.

# Default version, always set
default_version = '${_ext_VERSION}'

# Module pathname generated from target shared library name
module_pathname = '$libdir/$<TARGET_FILE_NAME:${NAME}>'
"
    )

    install(
        TARGETS ${NAME}
        LIBRARY DESTINATION ${PostgreSQL_PACKAGE_LIBRARY_DIR}
    )

    install(
        FILES ${_control_file} ${_ext_SCRIPTS}
        DESTINATION ${PostgreSQL_EXTENSION_DIR}
    )

    if(_ext_REGRESS)
        add_test(
            NAME ${NAME}_regress
            COMMAND ${PG_REGRESS}
                    --bindir=${_pg_bindir}
                    --dlpath=${PostgreSQL_PACKAGE_LIBRARY_DIR}
                    --inputdir=${CMAKE_SOURCE_DIR}/tests
                    --outputdir=${CMAKE_CURRENT_BINARY_DIR}
                    --load-extension=${NAME}
                    ${_ext_REGRESS}
        )
    endif()
endfunction()

function(add_mlir_unit_test NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRControlFlowDialect
        MLIRSCFDialect
        MLIRFuncToLLVM
        MLIRMathToLLVM
        MLIRSCFToControlFlow
        MLIRControlFlowToLLVM
        MLIRArithToLLVM
        MLIRTransforms
        MLIR
        LLVM
    )
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/src/dialects
        ${PostgreSQL_SERVER_INCLUDE_DIRS}
    )
    set_target_properties(${NAME} PROPERTIES
        INSTALL_RPATH "/usr/lib/llvm-20/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endfunction() 