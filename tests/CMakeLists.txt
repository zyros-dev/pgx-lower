# Top-level test CMakeLists.txt
# Include all unit test subdirectories

add_subdirectory(unit/conversion)
add_subdirectory(unit/dialects)
add_subdirectory(unit/execution)
add_subdirectory(unit/extension)
add_subdirectory(unit/mlir)
add_subdirectory(unit/runtime)