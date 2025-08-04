# MLIR Conversion Passes CMakeLists.txt
# Conversion passes handle lowering between dialects

add_subdirectory(RelAlgToSubOp)
add_subdirectory(SubOpToControlFlow)
add_subdirectory(DBToDSA)
add_subdirectory(DBToLLVM)
add_subdirectory(DSAToLLVM)
add_subdirectory(UtilToLLVM)

# Note: ArrowToStd and DBToStd directories exist but are empty
# Uncomment when implemented:
# add_subdirectory(ArrowToStd)
# add_subdirectory(DBToStd)