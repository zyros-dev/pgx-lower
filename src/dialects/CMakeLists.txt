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

# TODO: Enable TableGen when fixed
# set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgOps.td)
# mlir_tablegen(PgOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
# mlir_tablegen(PgOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
# add_public_tablegen_target(MLIRPgOpsIncGen)

# set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgTypes.td)
# mlir_tablegen(PgTypes.h.inc -gen-typedef-decls --typedefs-dialect=pg -I${MLIR_INCLUDE_DIRS})
# mlir_tablegen(PgTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=pg -I${MLIR_INCLUDE_DIRS})
# add_public_tablegen_target(MLIRPgTypesIncGen)

add_library(MLIRPgDialect STATIC
    pg/PgDialect.cpp
    pg/LowerPgToSCF.cpp
)

target_include_directories(MLIRPgDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/pg
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRPgDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRInferTypeOpInterface
    MLIRFunctionInterfaces
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRTransforms
)

# Ensure position independent code for shared library usage
set_target_properties(MLIRPgDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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

# Ensure position independent code for shared library usage
set_target_properties(PgxLowerRuntime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)