# Add subdirectories for each conversion pass
add_subdirectory(RelAlgToDB)
add_subdirectory(DBToStd)
add_subdirectory(DSAToLLVM)

# Additional conversion passes for complete pipeline:
# add_subdirectory(DSAToStd)     # DSA → Standard MLIR conversion
# add_subdirectory(UtilToLLVM)   # Util → LLVM IR conversion