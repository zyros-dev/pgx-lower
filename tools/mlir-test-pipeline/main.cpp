#include <iostream>
#include <string>

// MLIR Core Infrastructure  
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

// Our standalone MLIR pipeline components
#include "execution/mlir_pipeline_standalone.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_relalg_file.mlir>" << std::endl;
        return 1;
    }
    
    std::string inputFile = argv[1];
    
    // Setup MLIR context
    mlir::MLIRContext context;
    if (!mlir_standalone::setupMLIRContextForJIT(context)) {
        std::cerr << "Failed to setup MLIR context" << std::endl;
        return 1;
    }
    
    // Parse RelAlg module from file
    auto moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &context);
    if (!moduleRef) {
        std::cerr << "Failed to parse MLIR module from: " << inputFile << std::endl;
        return 1;
    }
    mlir::ModuleOp module = moduleRef.get();
    
    // Run complete lowering pipeline: RelAlg → DB+DSA+Util → Standard → LLVM
    if (!mlir_standalone::runCompleteLoweringPipeline(module)) {
        std::cerr << "Pipeline execution failed" << std::endl;
        return 1;
    }
    
    // Output the transformed module to stdout
    module.print(llvm::outs());
    
    return 0;
}