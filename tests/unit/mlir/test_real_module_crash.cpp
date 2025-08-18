#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "pgx_lower/mlir/Passes.h"
#include "pgx_lower/mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace mlir;

class RealModuleCrashTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        context.loadDialect<util::UtilDialect>();
    }

    MLIRContext context;
};

TEST_F(RealModuleCrashTest, ExactRealModuleContent) {
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
    %4 = util.varlen32_create_const "{ \"table\": \"test|oid:5741428\", \"columns\": [ \"dummy_col\"] }"
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
    ASSERT_TRUE(module) << "Failed to parse real module";
    
    PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    bool result = succeeded(pm.run(module.get()));
    
    if (result) {
        std::string outputStr;
        llvm::raw_string_ostream stream(outputStr);
        module->print(stream);
        
        std::ofstream outFile("/tmp/converted_llvm_ir.mlir");
        if (outFile.is_open()) {
            outFile << outputStr;
            outFile.close();
        } else {
        }
        
    }
    
    EXPECT_TRUE(result) << "StandardToLLVMPass failed on real module content";
}