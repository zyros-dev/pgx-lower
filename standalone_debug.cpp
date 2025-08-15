#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Passes.h"

int main() {
    mlir::MLIRContext context;
    
    // Load required dialects
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    context.loadDialect<mlir::util::UtilDialect>();
    
    const char* moduleStr = R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
})";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
    if (!module) {
        printf("Failed to parse module\n");
        return 1;
    }
    
    printf("Module parsed successfully\n");
    
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    printf("About to run StandardToLLVMPass...\n");
    
    if (mlir::succeeded(pm.run(module.get()))) {
        printf("SUCCESS: StandardToLLVMPass worked!\n");
        return 0;
    } else {
        printf("FAILED: StandardToLLVMPass failed\n");
        return 1;
    }
}