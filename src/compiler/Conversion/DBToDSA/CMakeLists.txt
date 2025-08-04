add_mlir_conversion_library(MLIRDBToDSA
        LowerDBToDSA.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDBDialect
        MLIRDSADialect
)

set_target_properties(MLIRDBToDSA PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)