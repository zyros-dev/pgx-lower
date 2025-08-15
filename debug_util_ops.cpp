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

void testOperation(MLIRContext& context, const std::string& name, const std::string& moduleStr) {
    std::cout << "Testing: " << name << std::endl;
    
    auto module = parseSourceString<ModuleOp>(moduleStr, &context);
    if (!module) {
        std::cout << "❌ Failed to parse: " << name << std::endl;
        return;
    }
    
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    if (succeeded(pm.run(module.get()))) {
        std::cout << "✅ SUCCESS: " << name << std::endl;
    } else {
        std::cout << "❌ FAILED: " << name << std::endl;
    }
}

int main() {
    MLIRContext context;
    
    // Load required dialects
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    context.loadDialect<LLVM::LLVMDialect>();
    context.loadDialect<util::UtilDialect>();
    
    // Test 1: Simple arithmetic only
    testOperation(context, "simple arithmetic", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
}
)");

    // Test 2: Add util.alloca
    testOperation(context, "util.alloca", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    return %c42_i32 : i32
  }
}
)");

    // Test 3: Add util.varlen32_create_const
    testOperation(context, "util.varlen32_create_const", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %1 = util.varlen32_create_const ""
    return %c42_i32 : i32
  }
}
)");

    // Test 4: Add util.generic_memref_cast  
    testOperation(context, "util.generic_memref_cast", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    %7 = util.generic_memref_cast %0 : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>> -> <i8>
    return %c42_i32 : i32
  }
}
)");

    // Test 5: Add util.tupleelementptr
    testOperation(context, "util.tupleelementptr", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    %8 = util.tupleelementptr %0[0] : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>> -> <index>
    return %c42_i32 : i32
  }
}
)");

    // Test 6: Add util.load
    testOperation(context, "util.load", R"(
module {
  func.func @main() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    %8 = util.tupleelementptr %0[0] : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>> -> <index>
    %9 = util.load %8[] : <index> -> index
    return %c42_i32 : i32
  }
}
)");
    
    return 0;
}