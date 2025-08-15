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
#include "pgx_lower/mlir/Passes.h"
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
    
    // The exact MLIR from /tmp/real_module.mlir
    const char* moduleStr = R"(
module {
  func.func private @_ZN7runtime12TableBuilder5buildEv(!util.ref<i8>) -> !util.ref<i8>
  func.func private @_ZN7runtime12TableBuilder7nextRowEv(!util.ref<i8>)
  func.func private @_ZN7runtime19DataSourceIteration3endEPS0_(!util.ref<i8>)
  func.func private @_ZN7runtime19DataSourceIteration4nextEv(!util.ref<i8>)
  func.func private @_ZN7runtime19DataSourceIteration6accessEPNS0_15RecordBatchInfoE(!util.ref<i8>, !util.ref<i8>)
  func.func private @_ZN7runtime19DataSourceIteration7isValidEv(!util.ref<i8>) -> i1
  func.func private @_ZN7runtime19DataSourceIteration5startEPNS_16ExecutionContextENS_8VarLen32E(!util.ref<i8>, !util.varlen32) -> !util.ref<i8>
  func.func private @rt_get_execution_context() -> !util.ref<i8>
  func.func private @_ZN7runtime12TableBuilder6createENS_8VarLen32E(!util.varlen32) -> !util.ref<i8>
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i32 = arith.constant 42 : i32
    %0 = util.alloca() : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>>
    %1 = util.varlen32_create_const ""
    %2 = call @_ZN7runtime12TableBuilder6createENS_8VarLen32E(%1) : (!util.varlen32) -> !util.ref<i8>
    %3 = call @rt_get_execution_context() : () -> !util.ref<i8>
    %4 = util.varlen32_create_const "{ \"table\": \"test|oid:5749620\", \"columns\": [ \"dummy_col\"] }"
    %5 = call @_ZN7runtime19DataSourceIteration5startEPNS_16ExecutionContextENS_8VarLen32E(%3, %4) : (!util.ref<i8>, !util.varlen32) -> !util.ref<i8>
    scf.while : () -> () {
      %7 = func.call @_ZN7runtime19DataSourceIteration7isValidEv(%5) : (!util.ref<i8>) -> i1
      scf.condition(%7)
    } do {
      %7 = util.generic_memref_cast %0 : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>> -> <i8>
      func.call @_ZN7runtime19DataSourceIteration6accessEPNS0_15RecordBatchInfoE(%5, %7) : (!util.ref<i8>, !util.ref<i8>) -> ()
      %8 = util.tupleelementptr %0[0] : <tuple<index, index, index, !util.ref<i8>, !util.ref<i32>, !util.ref<i8>>> -> <index>
      %9 = util.load %8[] : <index> -> index
      scf.for %arg0 = %c0 to %9 step %c1 {
        func.call @_ZN7runtime12TableBuilder7nextRowEv(%2) : (!util.ref<i8>) -> ()
      }
      func.call @_ZN7runtime19DataSourceIteration4nextEv(%5) : (!util.ref<i8>) -> ()
      scf.yield
    }
    call @_ZN7runtime19DataSourceIteration3endEPS0_(%5) : (!util.ref<i8>) -> ()
    %6 = call @_ZN7runtime12TableBuilder5buildEv(%2) : (!util.ref<i8>) -> !util.ref<i8>
    return %c42_i32 : i32
  }
}
)";

    auto module = parseSourceString<ModuleOp>(moduleStr, &context);
    if (!module) {
        std::cout << "Failed to parse module" << std::endl;
        return 1;
    }
    
    std::cout << "Module parsed successfully" << std::endl;
    
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    std::cout << "About to run StandardToLLVMPass..." << std::endl;
    
    if (succeeded(pm.run(module.get()))) {
        std::cout << "SUCCESS: StandardToLLVMPass worked!" << std::endl;
        
        // Dump the resulting LLVM IR
        std::string outputStr;
        llvm::raw_string_ostream stream(outputStr);
        module->print(stream);
        
        std::ofstream outFile("/tmp/converted_llvm_ir.mlir");
        if (outFile.is_open()) {
            outFile << outputStr;
            outFile.close();
            std::cout << "✅ Converted LLVM IR dumped to /tmp/converted_llvm_ir.mlir" << std::endl;
        } else {
            std::cout << "❌ Failed to dump converted IR" << std::endl;
        }
        
        // Print first 2000 chars to console
        std::cout << "\n=== CONVERTED LLVM IR (first 2000 chars) ===" << std::endl;
        std::cout << outputStr.substr(0, 2000) << std::endl;
        if (outputStr.length() > 2000) {
            std::cout << "...(truncated, see full output in /tmp/converted_llvm_ir.mlir)" << std::endl;
        }
        std::cout << "=== END CONVERTED IR ===" << std::endl;
        
        return 0;
    } else {
        std::cout << "FAILED: StandardToLLVMPass failed" << std::endl;
        return 1;
    }
}