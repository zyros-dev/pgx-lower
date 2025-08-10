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
add_subdirectory(../mlir/Dialect/Util UtilDialect)

# Add conversion passes - LingoDB-compliant parallel lowering architecture:
# RelAlg → (DB + DSA + Util) then parallel lowering to Standard MLIR → LLVM
add_subdirectory(../mlir/Conversion/RelAlgToDB RelAlgToDBConversion)
# DB → Standard MLIR conversion (PostgreSQL SPI integration)
add_subdirectory(../mlir/Conversion/DBToStd DBToStdConversion)
# DSA → Standard MLIR conversion (data structure operations)
add_subdirectory(../mlir/Conversion/DSAToStd DSAToStdConversion)
# DSA → LLVM conversion (data structure algorithms) - TODO: Import from LingoDB
# add_subdirectory(../mlir/Conversion/DSAToLLVM DSAToLLVMConversion)
# Util → LLVM conversion (utility operations and types)
add_subdirectory(../mlir/Conversion/UtilToLLVM UtilToLLVMConversion)

# Add centralized pass pipeline infrastructure
add_subdirectory(../mlir MLIRPasses)

# Architecture: RelAlg generates mixed (DB + DSA + Util) operations
# Parallel lowering: DB → Standard, DSA → LLVM, Util → LLVM

# Current Focus: Test 1 (SELECT * FROM test) through complete pipeline
# Architecture: PostgreSQL AST → RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM IR → JIT