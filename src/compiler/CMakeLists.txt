# Compiler infrastructure CMakeLists.txt
# Clean slate refactor - dialects and lowerings will be rebuilt incrementally

# Add frontend (when it exists)  
# add_subdirectory(frontend)

# Add MLIR support utilities (when they exists)
# add_subdirectory(mlir-support)

# NOTE: Dialect and Conversion directories removed for clean slate refactor
# Will be rebuilt incrementally using LingoDB 2022 architecture:
# RelAlg → DB → DSA → Standard MLIR