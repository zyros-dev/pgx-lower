add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(../Transforms Transforms)

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
    MLIRUtilToLLVM
    StandardToLLVM
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
)