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
include_directories(${CMAKE_SOURCE_DIR}/include/pgx_lower)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# PG dialect removed - using RelAlg instead

# SubOperator dialect TableGen - Fixed compatibility issues
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator/SubOpOps.td)
mlir_tablegen(SubOpDialect.h.inc -gen-dialect-decls -dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpDialect.cpp.inc -gen-dialect-defs -dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpTypes.h.inc -gen-typedef-decls --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOpsAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOpsAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOpsEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOpsEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
add_public_tablegen_target(MLIRSubOpIncGen)

# Generate interfaces separately from the interfaces file
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator/SubOpInterfaces.td)
mlir_tablegen(SubOpOpsInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpOpsInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpTypeInterfaces.h.inc -gen-type-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
mlir_tablegen(SubOpTypeInterfaces.cpp.inc -gen-type-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator)
add_public_tablegen_target(MLIRSubOpInterfacesIncGen)

# Database dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB/DBDialect.td)
mlir_tablegen(DBDialect.h.inc -gen-dialect-decls -dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DBDialect.cpp.inc -gen-dialect-defs -dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

# DB Operations require DBOps.td as target
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB/DBOps.td)
mlir_tablegen(DBOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/TupleStream -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg)
mlir_tablegen(DBOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/TupleStream -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg)
mlir_tablegen(DBOpsEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB)
mlir_tablegen(DBOpsEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB)

# DB Types are defined in DBOps.td
mlir_tablegen(DBTypes.h.inc -gen-typedef-decls --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DBTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

# DB Interfaces
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB/DBInterfaces.td)
mlir_tablegen(DBOpsInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DBOpsInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
add_public_tablegen_target(MLIRDBIncGen)

# TupleStream dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/TupleStream/TupleStreamBase.td)
mlir_tablegen(TupleStreamDialect.h.inc -gen-dialect-decls -dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamDialect.cpp.inc -gen-dialect-defs -dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamTypes.h.inc -gen-typedef-decls --typedefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/TupleStream/TupleStreamOps.td)
mlir_tablegen(TupleStreamOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(TupleStreamOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

add_public_tablegen_target(MLIRTupleStreamIncGen)

# DSA dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DSA/DSADialect.td)
mlir_tablegen(DSADialect.h.inc -gen-dialect-decls -dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DSADialect.cpp.inc -gen-dialect-defs -dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DSAOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DSAOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DSATypes.h.inc -gen-typedef-decls --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(DSATypes.cpp.inc -gen-typedef-defs --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
add_public_tablegen_target(MLIRDSAIncGen)

# Util dialect TableGen - MUST BE BEFORE SubOp since SubOp depends on Util
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/util/UtilBase.td)
mlir_tablegen(UtilDialect.h.inc -gen-dialect-decls -dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(UtilDialect.cpp.inc -gen-dialect-defs -dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(UtilTypes.h.inc -gen-typedef-decls --typedefs-dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(UtilTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
# Generate ops from UtilOps.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/util/UtilOps.td)
mlir_tablegen(UtilOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(UtilOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/util/UtilInterfaces.td)
mlir_tablegen(UtilOpsTypeInterfaces.h.inc -gen-type-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(UtilOpsTypeInterfaces.cpp.inc -gen-type-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
add_public_tablegen_target(MLIRUtilIncGen)

# RelAlg dialect TableGen - Enable for pure LingoDB architecture
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg/RelAlgBase.td)
mlir_tablegen(RelAlgDialect.h.inc -gen-dialect-decls -dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgDialect.cpp.inc -gen-dialect-defs -dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg/RelAlgInterfaces.td)
mlir_tablegen(RelAlgInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg/RelAlgOps.td)
mlir_tablegen(RelAlgOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg -I${CMAKE_SOURCE_DIR}/include/pgx_lower)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg/RelAlgTypes.td)
mlir_tablegen(RelAlgTypes.h.inc -gen-typedef-decls --typedefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
mlir_tablegen(RelAlgTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/pgx_lower)
add_public_tablegen_target(MLIRRelAlgIncGen)

# PG dialect library removed - using RelAlg instead

add_library(MLIRSubOpDialect STATIC
    SubOperator/SubOpDialect.cpp
    SubOperator/SubOpInterfaces.cpp
    SubOperator/SubOperatorOps.cpp
    # SubOp transform passes
    SubOperator/Transforms/Passes.cpp
    SubOperator/Transforms/FoldColumnsPass.cpp
    SubOperator/Transforms/EnforceOrder.cpp
    SubOperator/Transforms/NormalizeSubOpPass.cpp
    SubOperator/Transforms/SpecializeSubOpPass.cpp
    SubOperator/Transforms/PullGatherUpPass.cpp
    SubOperator/Transforms/ReuseLocalPass.cpp
    SubOperator/Transforms/GlobalOpt.cpp
    SubOperator/Transforms/ParallelizePass.cpp
    SubOperator/Transforms/SpecializeParallelPass.cpp
    SubOperator/Transforms/SplitIntoExecutionSteps.cpp
    SubOperator/Transforms/InlineNestedMapPass.cpp
    SubOperator/Transforms/FinalizePass.cpp
    SubOperator/Transforms/PrepareForLowering.cpp
    # Analysis classes
    SubOperator/Transforms/ColumnCreationAnalysis.cpp
    SubOperator/Transforms/ColumnUsageAnalysis.cpp
    SubOperator/Transforms/SubOpStateUsageTransformer.cpp
    SubOperator/Transforms/SubOpDependencyAnalysis.cpp
)

add_library(MLIRDBDialect STATIC
    DB/DBDialect.cpp
    DB/DBOps.cpp
    DB/DBTypes.cpp
)

add_library(MLIRTupleStreamDialect STATIC
    TupleStream/TupleStreamDialect.cpp
    TupleStream/ColumnManager.cpp
    TupleStream/TupleStreamOps.cpp
)

add_library(MLIRDSADialect STATIC
    DSA/DSADialect.cpp
)

add_library(MLIRRelAlgDialect STATIC
    RelAlg/RelAlgDialect.cpp
    RelAlg/RelAlgOps.cpp
)

add_library(MLIRUtilDialect STATIC
    util/UtilDialect.cpp
    util/FunctionHelper.cpp
    util/UtilOps.cpp
    util/UtilTypes.cpp
)

# Configure RelAlg dialect library
target_include_directories(MLIRRelAlgDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/RelAlg
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRRelAlgDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRDBDialect
    MLIRTupleStreamDialect
    MLIRArithDialect
)

set_target_properties(MLIRRelAlgDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Add dependency on generated files
# PG dialect dependencies removed
add_dependencies(MLIRSubOpDialect MLIRSubOpIncGen MLIRSubOpInterfacesIncGen MLIRTupleStreamIncGen MLIRUtilIncGen)
add_dependencies(MLIRDBDialect MLIRDBIncGen MLIRRelAlgIncGen)  # DBDialect needs RelAlg interfaces
add_dependencies(MLIRTupleStreamDialect MLIRTupleStreamIncGen)
add_dependencies(MLIRDSADialect MLIRDSAIncGen)
add_dependencies(MLIRRelAlgDialect MLIRRelAlgIncGen)
add_dependencies(MLIRUtilDialect MLIRUtilIncGen)

# PG dialect configuration removed

# Configure SubOperator dialect library
target_include_directories(MLIRSubOpDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/SubOperator
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src/compiler/Conversion/SubOpToControlFlow/Headers  # Add this line
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
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
    MLIRTupleStreamDialect
    MLIRSCFDialect
)

set_target_properties(MLIRSubOpDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure Database dialect library
target_include_directories(MLIRDBDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DB
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRDBDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRInferTypeOpInterface
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)

set_target_properties(MLIRDBDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure TupleStream dialect library
target_include_directories(MLIRTupleStreamDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/TupleStream
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRTupleStreamDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
)

set_target_properties(MLIRTupleStreamDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure DSA dialect library
target_include_directories(MLIRDSADialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/DSA
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
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

# Configure Util dialect library
target_include_directories(MLIRUtilDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/compiler/Dialect/util
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/compiler
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRUtilDialect PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRLLVMDialect
    MLIRArithDialect
    MLIRFuncDialect
)

set_target_properties(MLIRUtilDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)


add_library(PgxLowerRuntime
    ../../runtime/tuple_access.cpp
    ../../runtime/PostgreSQLDataSource.cpp
    ../../runtime/helpers.cpp
    ../../runtime/postgresql_runtime.cpp
)

target_include_directories(PgxLowerRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
)

target_compile_definitions(PgxLowerRuntime PRIVATE POSTGRESQL_EXTENSION)

target_link_libraries(PgxLowerRuntime
    ${PostgreSQL_LIBRARIES}
)

# Ensure position independent code for shared library usage
set_target_properties(PgxLowerRuntime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)