#include "compiler/runtime/helpers.h"
#include "compiler/Dialect/util/FunctionHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace pgx_lower::compiler::runtime {

std::vector<mlir::Value> RuntimeCallGenerator::operator()(const std::vector<mlir::Value>& args) {
    // Get the parent module
    auto module = builder.getBlock()->getParentOp();
    while (module && !mlir::isa<mlir::ModuleOp>(module)) {
        module = module->getParentOp();
    }
    
    if (!module) {
        // Fallback: create dummy values for now
        // TODO Phase 5: Proper error handling
        std::vector<mlir::Value> results;
        results.push_back(builder.create<mlir::LLVM::ZeroOp>(loc, 
            mlir::LLVM::LLVMPointerType::get(builder.getContext())));
        return results;
    }
    
    auto moduleOp = mlir::cast<mlir::ModuleOp>(module);
    
    // Look up or create the function
    auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(functionName);
    if (!funcOp) {
        // Create function signature based on common patterns
        llvm::SmallVector<mlir::Type> argTypes;
        for (auto arg : args) {
            argTypes.push_back(arg.getType());
        }
        
        // Most runtime functions return a pointer
        llvm::SmallVector<mlir::Type> resultTypes;
        resultTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));
        
        auto funcType = builder.getFunctionType(argTypes, resultTypes);
        
        // Insert at module level
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());
        funcOp = builder.create<mlir::func::FuncOp>(loc, functionName, funcType);
        funcOp.setPrivate();
    }
    
    // Create the call
    auto callOp = builder.create<mlir::func::CallOp>(loc, funcOp, args);
    return std::vector<mlir::Value>(callOp.getResults().begin(), callOp.getResults().end());
}

} // namespace pgx_lower::compiler::runtime