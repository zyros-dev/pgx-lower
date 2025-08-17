# Unit test subdirectories

add_subdirectory(mlir)
# add_subdirectory(postgres)  # Temporarily disabled due to target name conflicts
add_subdirectory(frontend)  # Re-enabled with split test files
add_subdirectory(execution)  # JIT execution engine tests