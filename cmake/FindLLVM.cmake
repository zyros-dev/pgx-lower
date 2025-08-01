# FindLLVM.cmake
# Finds the LLVM and MLIR packages

set(LLVM_DIR "/usr/lib/llvm-20/lib/cmake/llvm")
set(MLIR_DIR "/usr/lib/llvm-20/lib/cmake/mlir")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM and MLIR include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add LLVM and MLIR definitions
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS}) 