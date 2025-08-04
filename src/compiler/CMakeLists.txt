# Compiler infrastructure CMakeLists.txt
# This includes MLIR dialects and conversion passes

# Add dialect libraries
add_subdirectory(Dialect)

# Add conversion passes
add_subdirectory(Conversion)

# Add frontend (when it exists)  
# add_subdirectory(frontend)

# Add MLIR support utilities (when they exist)
# add_subdirectory(mlir-support)