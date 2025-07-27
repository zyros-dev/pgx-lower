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

# TableGen code generation for PostgreSQL dialect
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgDataAccess.td)
mlir_tablegen(PgDataAccess.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(PgDataAccess.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRPgDataAccessIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgPolymorphic.td)
mlir_tablegen(PgPolymorphic.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(PgPolymorphic.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRPgPolymorphicIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/pg/PgTypes.td)
mlir_tablegen(PgTypes.h.inc -gen-typedef-decls --typedefs-dialect=pg -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(PgTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=pg -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRPgTypesIncGen)

# SubOperator dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/subop/SubOpDialect.td)
mlir_tablegen(SubOpDialect.h.inc -gen-dialect-decls -dialect=subop -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(SubOpDialect.cpp.inc -gen-dialect-defs -dialect=subop -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(SubOpOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(SubOpOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(SubOpTypes.h.inc -gen-typedef-decls --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(SubOpTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRSubOpIncGen)

# Database dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/db/DBDialect.td)
mlir_tablegen(DBDialect.h.inc -gen-dialect-decls -dialect=db -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DBDialect.cpp.inc -gen-dialect-defs -dialect=db -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DBOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DBOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DBTypes.h.inc -gen-typedef-decls --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DBTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRDBIncGen)

# DSA dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/dsa/DSADialect.td)
mlir_tablegen(DSADialect.h.inc -gen-dialect-decls -dialect=dsa -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DSADialect.cpp.inc -gen-dialect-defs -dialect=dsa -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DSAOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DSAOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DSATypes.h.inc -gen-typedef-decls --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS})
mlir_tablegen(DSATypes.cpp.inc -gen-typedef-defs --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(MLIRDSAIncGen)

add_library(MLIRPgDialect STATIC
    pg/PgDialect.cpp
    pg/LowerPgToSCF.cpp
    pg/LowerPgToSubOp.cpp
    pg/PgPolymorphicOps.cpp
)

add_library(MLIRSubOpDialect STATIC
    subop/SubOpDialect.cpp
    subop/LowerSubOpToDB.cpp
)

add_library(MLIRDBDialect STATIC
    db/DBDialect.cpp
)

add_library(MLIRDSADialect STATIC
    dsa/DSADialect.cpp
)

# Add dependency on generated files
add_dependencies(MLIRPgDialect MLIRPgDataAccessIncGen MLIRPgTypesIncGen MLIRPgPolymorphicIncGen)
add_dependencies(MLIRSubOpDialect MLIRSubOpIncGen)
add_dependencies(MLIRDBDialect MLIRDBIncGen)
add_dependencies(MLIRDSADialect MLIRDSAIncGen)

target_include_directories(MLIRPgDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/pg
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
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
    MLIRSubOpDialect
)

# Ensure position independent code for shared library usage
set_target_properties(MLIRPgDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure SubOperator dialect library
target_include_directories(MLIRSubOpDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/subop
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRSubOpDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRCallInterfaces
    MLIRFunctionInterfaces
    MLIRDBDialect
    MLIRSCFDialect
)

set_target_properties(MLIRSubOpDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure Database dialect library
target_include_directories(MLIRDBDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/db
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRDBDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRInferTypeOpInterface
)

set_target_properties(MLIRDBDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure DSA dialect library
target_include_directories(MLIRDSADialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/dsa
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRDSADialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRLLVMDialect
)

set_target_properties(MLIRDSADialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(PgxLowerRuntime
    ../runtime/tuple_access.cpp
)

target_include_directories(PgxLowerRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(PgxLowerRuntime PRIVATE POSTGRESQL_EXTENSION)

target_link_libraries(PgxLowerRuntime
    ${PostgreSQL_LIBRARIES}
)

# Ensure position independent code for shared library usage
set_target_properties(PgxLowerRuntime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)