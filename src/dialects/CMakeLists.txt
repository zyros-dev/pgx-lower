# PostgreSQL Dialect Library

# Find MLIR components
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgOps.td)
mlir_tablegen(PgOps.h.inc -gen-op-decls)
mlir_tablegen(PgOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPgOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgTypes.td)
mlir_tablegen(PgTypes.h.inc -gen-typedef-decls --typedefs-dialect=pg)
mlir_tablegen(PgTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=pg)
add_public_tablegen_target(MLIRPgTypesIncGen)

add_mlir_library(MLIRPgDialect
    pg/PgDialect.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/include/dialects/pg
    
    DEPENDS
    MLIRPgOpsIncGen
    MLIRPgTypesIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRInferTypeOpInterface
    MLIRFunctionInterfaces
)

add_library(PgxLowerRuntime
    ../runtime/tuple_access.cpp
)

target_include_directories(PgxLowerRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(PgxLowerRuntime
    ${PostgreSQL_LIBRARIES}
)