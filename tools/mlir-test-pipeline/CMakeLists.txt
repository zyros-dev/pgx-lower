# MLIR Pipeline Test Tool
add_executable(mlir-test-pipeline
    main.cpp
    mlir_pipeline_standalone.cpp
)

# Update compile flags for LLVM compatibility
llvm_update_compile_flags(mlir-test-pipeline)

# Link with necessary libraries - using pattern from existing tools
target_link_libraries(mlir-test-pipeline
    PUBLIC
    # MLIR Passes and Dialects
    MLIRPasses
    
    # Standard MLIR Core
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRTransforms
    MLIRPass
    
    # Standard MLIR Dialects (using correct full paths)
    /usr/lib/llvm-20/lib/libMLIRFuncDialect.a
    /usr/lib/llvm-20/lib/libMLIRArithDialect.a
    /usr/lib/llvm-20/lib/libMLIRLLVMDialect.a
    /usr/lib/llvm-20/lib/libMLIRControlFlowDialect.a
    /usr/lib/llvm-20/lib/libMLIRMemRefDialect.a
    /usr/lib/llvm-20/lib/libMLIRSCFDialect.a
    
    # Command line parsing
    LLVMSupport
    LLVMCore
)

# Add compile definitions for building without PostgreSQL
target_compile_definitions(mlir-test-pipeline PRIVATE
    BUILDING_UNIT_TESTS=1
)

# Set C++20 standard
set_property(TARGET mlir-test-pipeline PROPERTY CXX_STANDARD 20)
set_property(TARGET mlir-test-pipeline PROPERTY CXX_STANDARD_REQUIRED ON)

# Set output directory to build root like other tools
set_target_properties(mlir-test-pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")