add_subdirectory(Dialect)
add_subdirectory(Conversion)

add_library(MLIRPasses
        Passes.cpp
)

target_link_libraries(MLIRPasses
    PUBLIC
    MLIRIR
    MLIRPass
    MLIRRelAlgToDB
    MLIRDBToArrowStd
    MLIRDSAToStd
    UtilToLLVM
    StandardToLLVM
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
)