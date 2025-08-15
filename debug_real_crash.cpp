#include <iostream>
#include <fstream>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h" 
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main() {
    MLIRContext context;
    
    // Load required dialects
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    context.loadDialect<LLVM::LLVMDialect>();
    context.loadDialect<util::UtilDialect>();
    
    // First test with a simple util operation to isolate the issue
    const char* simpleModuleStr = R"(
module {
  func.func @simple_test() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    return %c42_i32 : i32
  }
}
)";

    std::cout << "Testing simple module with just util.alloca..." << std::endl;
    
    auto simpleModule = parseSourceString<ModuleOp>(simpleModuleStr, &context);
    if (!simpleModule) {
        std::cout << "Failed to parse simple module" << std::endl;
        return 1;
    }
    
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    if (succeeded(pm.run(simpleModule.get()))) {
        std::cout << "✅ SUCCESS: Simple module with util.alloca converted" << std::endl;
    } else {
        std::cout << "❌ FAILED: Simple module conversion failed" << std::endl;
        return 1;
    }
    
    // Now test incrementally adding operations
    const char* incrementalModuleStr = R"(
module {
  func.func @incremental_test() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    %1 = util.varlen32_create_const ""
    return %c42_i32 : i32
  }
}
)";

    std::cout << "Testing incremental module with util.alloca + util.varlen32_create_const..." << std::endl;
    
    auto incrementalModule = parseSourceString<ModuleOp>(incrementalModuleStr, &context);
    if (!incrementalModule) {
        std::cout << "Failed to parse incremental module" << std::endl;
        return 1;
    }
    
    PassManager pm2(&context);
    pm2.addPass(pgx_lower::createStandardToLLVMPass());
    
    if (succeeded(pm2.run(incrementalModule.get()))) {
        std::cout << "✅ SUCCESS: Incremental module converted" << std::endl;
    } else {
        std::cout << "❌ FAILED: Incremental module conversion failed" << std::endl;
        std::cout << "The issue is likely with util.varlen32_create_const" << std::endl;
        return 1;
    }
    
    return 0;
}