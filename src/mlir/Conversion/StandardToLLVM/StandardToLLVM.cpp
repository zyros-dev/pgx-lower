#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "execution/logging.h"

namespace mlir {
namespace pgx_lower {

namespace {

struct StandardToLLVMPass : public PassWrapper<StandardToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandardToLLVMPass)

    void getDependentDialects(DialectRegistry& registry) const override {
        PGX_INFO("StandardToLLVMPass: Get dialects");
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect,
                       cf::ControlFlowDialect, arith::ArithDialect>();
    }

    void runOnOperation() override {
        PGX_INFO("[StandardToLLVMPass] Starting run on operation!");
        // The first thing to define is the conversion target. This will define the
        // final target for this lowering. For this lowering, we are only targeting
        // the LLVM dialect.
        const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();

        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp>();

        // During this lowering, we will also be lowering the MemRef types, that are
        // currently being operated on, to a representation in LLVM. To perform this
        // conversion we use a TypeConverter as part of the lowering. This converter
        // details how one type maps to another. This is necessary now that we will be
        // doing more complicated lowerings, involving loop region arguments.
        PGX_INFO("[StandardToLLVMPass] Init options");
        mlir::LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(getOperation()));
        // options.emitCWrappers = true;
        mlir::LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
        typeConverter.addSourceMaterialization(
            [&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
                return valueRange.front();
            });

        PGX_INFO("[StandardToLLVMPass] registering!");
        mlir::RewritePatternSet patterns(&getContext());
        populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

        mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        // We want to completely lower to LLVM, so we use a `FullConversion`. This
        // ensures that only legal operations will remain after the conversion.
        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createStandardToLLVMPass() {
    PGX_INFO("Creating StandardToLLVMPass instance");
    return std::make_unique<StandardToLLVMPass>();
}

} // namespace pgx_lower
} // namespace mlir