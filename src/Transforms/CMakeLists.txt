set(LLVM_TARGET_DEFINITIONS CustomCanonicalization.td)
mlir_tablegen(CustomCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRCustomCanonicalizationIncGen)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lingo-db/include)
add_mlir_library(MLIRCustomTransforms
        SinkOp.cpp
        SimplifyMemrefs.cpp
SimplifyArithmetics.cpp
        DEPENDS
        MLIRCustomCanonicalizationIncGen

        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRCopyOpInterface
        MLIRLoopLikeInterface
        MLIRPass
        MLIRSupport
        MLIRTransformUtils
        )