# Compiler infrastructure CMakeLists.txt
# Clean slate refactor - dialects and lowerings will be rebuilt incrementally

# Add frontend (when it exists)  
# add_subdirectory(frontend)

# Add MLIR support utilities (when they exists)
# add_subdirectory(mlir-support)

# Add MLIR dialects - starting with RelAlg
add_subdirectory(../mlir/Dialect/RelAlg RelAlgDialect)
add_subdirectory(../mlir/Dialect/DB DBDialect)
add_subdirectory(../mlir/Dialect/DSA DSADialect)

# Add conversion passes
add_subdirectory(../mlir/Conversion/RelAlgToDSA RelAlgToDSAConversion)

# NOTE: Remaining dialects and conversions will be rebuilt incrementally
# Architecture: RelAlg → DB → DSA → Standard MLIR