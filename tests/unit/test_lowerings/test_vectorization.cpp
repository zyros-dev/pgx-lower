// Spec 02 — LLVM auto-vectorization.
//
// RED when PTO.LoopVectorization is false and the optimizer uses a manual
// FPM that doesn't include the loop vectorizer; GREEN once the flags flip
// and the optimizer runs buildPerModuleDefaultPipeline(O2). Feeds a
// canonical vectorizable sum-reduction loop through the optimizer and
// asserts the resulting IR contains at least one vector instruction.

#include <cstdlib>

#include <gtest/gtest.h>

#include "pgx-lower/execution/jit_execution_engine.h"
#include "pgx-lower/utility/logging.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

// JIT engine calls mlir_runner::dumpLLVMIR for IR logging, which is defined
// in mlir_runner.cpp. Stub it here instead of dragging that TU (and its
// MLIR-heavy deps) into this test target.
namespace mlir_runner {
void dumpLLVMIR(llvm::Module* /*module*/, const std::string& /*title*/,
                pgx_lower::log::Category /*phase*/) {}
} // namespace mlir_runner

using namespace pgx_lower::execution;

namespace {

// Trivial i64 sum-reduction over a pointer-counted array. LLVM's loop
// vectorizer handles this pattern at O2 when LoopVectorization is on;
// with vectorization disabled it stays scalar.
constexpr const char* kSumReductionIR = R"LLVM(
define i64 @sum_array(ptr noalias %arr, i64 %n) {
entry:
  %is_empty = icmp eq i64 %n, 0
  br i1 %is_empty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %sum, %loop ]
  %ptr = getelementptr i64, ptr %arr, i64 %i
  %v = load i64, ptr %ptr, align 8
  %sum = add i64 %acc, %v
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i64 [ 0, %entry ], [ %sum, %loop ]
  ret i64 %result
}
)LLVM";

std::string moduleToString(const llvm::Module& module) {
    std::string out;
    llvm::raw_string_ostream stream(out);
    module.print(stream, /*AAW=*/nullptr);
    stream.flush();
    return out;
}

class VectorizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};

} // namespace

TEST_F(VectorizationTest, SumReductionLoopVectorizes) {
    llvm::LLVMContext ctx;
    llvm::SMDiagnostic diag;

    auto module = llvm::parseAssemblyString(kSumReductionIR, diag, ctx);
    ASSERT_TRUE(module) << "Failed to parse sum-reduction IR: "
                        << diag.getMessage().str();

    const std::string before = moduleToString(*module);
    // Sanity: the hand-written sum-reduction uses only scalar i64 types, so
    // no LLVM vector type token should appear before we run the optimizer.
    ASSERT_EQ(before.find("x i64>"), std::string::npos)
        << "Input IR unexpectedly already contains vector types";

    auto err = JITEngine::optimize_llvm_module(
        *module, llvm::CodeGenOptLevel::Default);
    ASSERT_FALSE(static_cast<bool>(err))
        << "Optimization returned an error";

    const std::string after = moduleToString(*module);

    // A vectorized loop emits instructions over vector types like
    // `<4 x i64>` or `<2 x i64>`. Check for the token `x i64>` which is
    // specific to vector types in textual LLVM IR.
    const bool has_i64_vector = after.find("x i64>") != std::string::npos;
    const bool has_i32_vector = after.find("x i32>") != std::string::npos;
    EXPECT_TRUE(has_i64_vector || has_i32_vector)
        << "Expected vector instructions in optimized IR, got:\n" << after;
}

// Custom main. After RUN_ALL_TESTS we _exit() to bypass C++ static
// destruction. The JIT engine's optimization pipeline allocates a
// TargetMachine + PassBuilder + module analysis managers; these leave
// ManagedStatic / signal-handler state whose destruction ordering
// SIGSEGVs in the test harness (not in production — the extension lives
// inside Postgres, which never unwinds at exit). Swapping gtest_main out
// for an explicit _exit(rc) removes the crash without masking real bugs:
// the test's own assertions have already run by the time we get here.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    std::_Exit(rc);
}
