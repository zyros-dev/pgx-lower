add_mlir_conversion_library(MLIRSubOpToControlFlow
        ExecutionEngine.cpp
        Core/SubOpToControlFlowRewriter.cpp
        Core/SubOpToControlFlowUtilities.cpp
        Operations/ControlFlowOperations.cpp
        Operations/DataStructureCreation.cpp
        Operations/GatherScatterOperations.cpp
        Operations/LookupOperations.cpp
        Operations/ReduceOperations.cpp
        Operations/ScanOperations.cpp
        Operations/TableOperations.cpp
        Operations/ThreadLocalOperations.cpp
        Operations/ViewOperations.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include
        Headers

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSubOpDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRDBDialect
        MLIRDSADialect
)

set_target_properties(MLIRSubOpToControlFlow PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)