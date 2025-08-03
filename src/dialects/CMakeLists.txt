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

# PG dialect removed - using RelAlg instead

# SubOperator dialect TableGen - Fixed compatibility issues
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/subop/SubOpOps.td)
mlir_tablegen(SubOpDialect.h.inc -gen-dialect-decls -dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpDialect.cpp.inc -gen-dialect-defs -dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpTypes.h.inc -gen-typedef-decls --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOpsAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOpsAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect=subop -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOpsEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOpsEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
add_public_tablegen_target(MLIRSubOpIncGen)

# Generate interfaces separately from the interfaces file
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/subop/SubOpInterfaces.td)
mlir_tablegen(SubOpOpsInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpOpsInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpTypeInterfaces.h.inc -gen-type-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
mlir_tablegen(SubOpTypeInterfaces.cpp.inc -gen-type-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/subop)
add_public_tablegen_target(MLIRSubOpInterfacesIncGen)

# Database dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/db/DBDialect.td)
mlir_tablegen(DBDialect.h.inc -gen-dialect-decls -dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DBDialect.cpp.inc -gen-dialect-defs -dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

# DB Operations require DBOps.td as target
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/db/DBOps.td)
mlir_tablegen(DBOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/db -I${CMAKE_SOURCE_DIR}/include/dialects/tuplestream -I${CMAKE_SOURCE_DIR}/include/dialects/relalg)
mlir_tablegen(DBOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/db -I${CMAKE_SOURCE_DIR}/include/dialects/tuplestream -I${CMAKE_SOURCE_DIR}/include/dialects/relalg)
mlir_tablegen(DBOpsEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/db)
mlir_tablegen(DBOpsEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/db)

# DB Types are defined in DBOps.td
mlir_tablegen(DBTypes.h.inc -gen-typedef-decls --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DBTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=db -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

# DB Interfaces
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/db/DBInterfaces.td)
mlir_tablegen(DBOpsInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DBOpsInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRDBIncGen)

# TupleStream dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/tuplestream/TupleStreamBase.td)
mlir_tablegen(TupleStreamDialect.h.inc -gen-dialect-decls -dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamDialect.cpp.inc -gen-dialect-defs -dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamTypes.h.inc -gen-typedef-decls --typedefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=tuples -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/tuplestream/TupleStreamOps.td)
mlir_tablegen(TupleStreamOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(TupleStreamOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

add_public_tablegen_target(MLIRTupleStreamIncGen)

# DSA dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/dsa/DSADialect.td)
mlir_tablegen(DSADialect.h.inc -gen-dialect-decls -dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DSADialect.cpp.inc -gen-dialect-defs -dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DSAOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DSAOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DSATypes.h.inc -gen-typedef-decls --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(DSATypes.cpp.inc -gen-typedef-defs --typedefs-dialect=dsa -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRDSAIncGen)

# Util dialect TableGen - MUST BE BEFORE SubOp since SubOp depends on Util
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/util/UtilBase.td)
mlir_tablegen(UtilDialect.h.inc -gen-dialect-decls -dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(UtilDialect.cpp.inc -gen-dialect-defs -dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(UtilTypes.h.inc -gen-typedef-decls --typedefs-dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(UtilTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=util -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
# Generate ops from UtilOps.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/util/UtilOps.td)
mlir_tablegen(UtilOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(UtilOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/util/UtilInterfaces.td)
mlir_tablegen(UtilOpsTypeInterfaces.h.inc -gen-type-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(UtilOpsTypeInterfaces.cpp.inc -gen-type-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRUtilIncGen)

# RelAlg dialect TableGen - Enable for pure LingoDB architecture
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/relalg/RelAlgBase.td)
mlir_tablegen(RelAlgDialect.h.inc -gen-dialect-decls -dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgDialect.cpp.inc -gen-dialect-defs -dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/relalg/RelAlgInterfaces.td)
mlir_tablegen(RelAlgInterfaces.h.inc -gen-op-interface-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgInterfaces.cpp.inc -gen-op-interface-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/relalg/RelAlgOps.td)
mlir_tablegen(RelAlgOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgEnums.h.inc -gen-enum-decls -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgEnums.cpp.inc -gen-enum-defs -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include/dialects/relalg -I${CMAKE_SOURCE_DIR}/include)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/dialects/relalg/RelAlgTypes.td)
mlir_tablegen(RelAlgTypes.h.inc -gen-typedef-decls --typedefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(RelAlgTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=relalg -I${MLIR_INCLUDE_DIRS} -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRRelAlgIncGen)

# PG dialect library removed - using RelAlg instead

add_library(MLIRSubOpDialect STATIC
    subop/SubOpDialect.cpp
    subop/SubOpInterfaces.cpp
    subop/SubOperatorOps.cpp
    # SubOpToControlFlow split files
    subop/SubOpToControlFlow/Core/SubOpToControlFlowUtilities.cpp
    subop/SubOpToControlFlow/Core/SubOpToControlFlowRewriter.cpp
    subop/SubOpToControlFlow/Operations/TableOperations.cpp
    subop/SubOpToControlFlow/Operations/ScanOperations.cpp
    subop/SubOpToControlFlow/Operations/LookupOperations.cpp
    subop/SubOpToControlFlow/Operations/DataStructureCreation.cpp
    subop/SubOpToControlFlow/Operations/GatherScatterOperations.cpp
    subop/SubOpToControlFlow/Operations/ReduceOperations.cpp
    subop/SubOpToControlFlow/Operations/ViewOperations.cpp
    subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp
    subop/SubOpToControlFlow/Operations/ThreadLocalOperations.cpp
    subop/SubOpToControlFlow/ExecutionEngine.cpp
    subop/LowerSubOpToDB.cpp
    # SubOp transform passes
    subop/Transforms/Passes.cpp
    subop/Transforms/FoldColumnsPass.cpp
    subop/Transforms/EnforceOrder.cpp
    subop/Transforms/NormalizeSubOpPass.cpp
    subop/Transforms/SpecializeSubOpPass.cpp
    subop/Transforms/PullGatherUpPass.cpp
    subop/Transforms/ReuseLocalPass.cpp
    subop/Transforms/GlobalOpt.cpp
    subop/Transforms/ParallelizePass.cpp
    subop/Transforms/SpecializeParallelPass.cpp
    subop/Transforms/SplitIntoExecutionSteps.cpp
    subop/Transforms/InlineNestedMapPass.cpp
    subop/Transforms/FinalizePass.cpp
    subop/Transforms/PrepareForLowering.cpp
    # Analysis classes
    subop/Transforms/ColumnCreationAnalysis.cpp
    subop/Transforms/ColumnUsageAnalysis.cpp
    subop/Transforms/SubOpStateUsageTransformer.cpp
    subop/Transforms/SubOpDependencyAnalysis.cpp
)

add_library(MLIRDBDialect STATIC
    db/DBDialect.cpp
    db/DBOps.cpp
    db/DBTypes.cpp
    db/LowerDBToDSA.cpp
    db/LowerDBToLLVM.cpp
)

add_library(MLIRTupleStreamDialect STATIC
    tuplestream/TupleStreamDialect.cpp
    tuplestream/ColumnManager.cpp
    tuplestream/TupleStreamOps.cpp
)

add_library(MLIRDSADialect STATIC
    dsa/DSADialect.cpp
)

add_library(MLIRRelAlgDialect STATIC
    relalg/RelAlgDialect.cpp
    relalg/RelAlgOps.cpp
    relalg/LowerRelAlgToSubOp.cpp
)

add_library(MLIRUtilDialect STATIC
    util/UtilDialect.cpp
    util/FunctionHelper.cpp
    util/UtilOps.cpp
    util/UtilTypes.cpp
    util/LowerDSAToLLVM.cpp
    relalg/Conversion/UtilToLLVM/UtilTypeConversion.cpp
)

# Configure RelAlg dialect library
target_include_directories(MLIRRelAlgDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/relalg
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
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
    ${CMAKE_SOURCE_DIR}/include/dialects/subop
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/dialects/subop/SubOpToControlFlow/Headers  # Add this line
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
    MLIRTupleStreamDialect
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
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)

set_target_properties(MLIRDBDialect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Configure TupleStream dialect library
target_include_directories(MLIRTupleStreamDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/tuplestream
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
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

# Configure Util dialect library
target_include_directories(MLIRUtilDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include/dialects/util
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
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
    ../runtime/tuple_access.cpp
    ../runtime/PostgreSQLDataSource.cpp
    ../runtime/helpers.cpp
    ../runtime/postgresql_runtime.cpp
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