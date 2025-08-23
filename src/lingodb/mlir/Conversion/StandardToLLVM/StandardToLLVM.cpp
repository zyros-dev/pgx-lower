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

class MainFunctionVisibilityPattern : public OpConversionPattern<func::FuncOp> {
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        if (funcOp.getSymName() != "main") {
            return failure();
        }

        auto convertedType = getTypeConverter()->convertType(funcOp.getFunctionType());
        auto funcType = mlir::dyn_cast<LLVM::LLVMFunctionType>(convertedType);
        if (!funcType) {
            return failure();
        }

        auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
            funcOp.getLoc(), funcOp.getSymName(), funcType);
        
        llvmFunc.setSymVisibilityAttr(rewriter.getStringAttr("public"));

        rewriter.inlineRegionBefore(funcOp.getBody(), llvmFunc.getBody(), llvmFunc.end());

        rewriter.eraseOp(funcOp);
        return success();
    }
};

struct StandardToLLVMPass : public PassWrapper<StandardToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandardToLLVMPass)

    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect,
                       cf::ControlFlowDialect, arith::ArithDialect>();
    }

    void runOnOperation() override {
        const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();

        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(getOperation()));
        mlir::LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
        typeConverter.addSourceMaterialization(
            [&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
                return valueRange.front();
            });

        mlir::RewritePatternSet patterns(&getContext());
        
        patterns.add<MainFunctionVisibilityPattern>(typeConverter, &getContext());
        
        populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
        
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        
        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        } else {
            module.walk([&](LLVM::LLVMFuncOp func) {
                if (func.isDeclaration()) {
                    func.setLinkageAttr(LLVM::LinkageAttr::get(&getContext(), LLVM::Linkage::External));
                    func.setSymVisibilityAttr(mlir::StringAttr::get(&getContext(), "default"));
                } else {
                    func.setSymVisibilityAttr(mlir::StringAttr::get(&getContext(), "public"));
                }
            });
        }
    }
};

} // namespace

std::unique_ptr<Pass> createStandardToLLVMPass() {
    return std::make_unique<StandardToLLVMPass>();
}

} // namespace pgx_lower
} // namespace mlir