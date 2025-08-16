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

// Custom pattern to ensure main function has proper visibility for ExecutionEngine
class MainFunctionVisibilityPattern : public OpConversionPattern<func::FuncOp> {
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        PGX_INFO("🔍 CustomMainPattern: Checking function: " + funcOp.getSymName().str());
        
        // Only handle main function - let default patterns handle others for proper type conversion
        if (funcOp.getSymName() != "main") {
            PGX_INFO("❌ Not main function, letting default pattern handle: " + funcOp.getSymName().str());
            return failure(); // Let default pattern handle this
        }

        PGX_INFO("🎯 MAIN FUNCTION FOUND! Applying custom PUBLIC visibility");

        // Convert function type using the same converter as default patterns
        auto convertedType = getTypeConverter()->convertType(funcOp.getFunctionType());
        auto funcType = mlir::dyn_cast<LLVM::LLVMFunctionType>(convertedType);
        if (!funcType) {
            PGX_ERROR("❌ Failed to convert main function type");
            return failure();
        }

        // Create LLVM function with PUBLIC visibility
        auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
            funcOp.getLoc(), funcOp.getSymName(), funcType);
        
        // Set PUBLIC visibility for ExecutionEngine lookup
        llvmFunc.setSymVisibilityAttr(rewriter.getStringAttr("public"));
        PGX_INFO("✅ Set sym_visibility=\"public\" on main function for ExecutionEngine");

        // Move function body
        rewriter.inlineRegionBefore(funcOp.getBody(), llvmFunc.getBody(), llvmFunc.end());

        rewriter.eraseOp(funcOp);
        PGX_INFO("🎉 Successfully converted main function with PUBLIC visibility");
        return success();
    }
};

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
        
        PGX_INFO("🔧 Adding MainFunctionVisibilityPattern to pattern set");
        patterns.add<MainFunctionVisibilityPattern>(typeConverter, &getContext());
        PGX_INFO("✅ MainFunctionVisibilityPattern added successfully");
        
        populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
        
        PGX_INFO("🔄 ADDING default func patterns for type conversion, but custom pattern will override main");
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        // We want to completely lower to LLVM, so we use a `FullConversion`. This
        // ensures that only legal operations will remain after the conversion.
        auto module = getOperation();
        PGX_INFO("🚀 About to run applyFullConversion with custom patterns");
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("❌ applyFullConversion failed!");
            signalPassFailure();
        } else {
            PGX_INFO("✅ applyFullConversion succeeded!");
            
            // POST-PROCESS: Fix main function visibility for ExecutionEngine lookup
            PGX_INFO("🔧 Post-processing: Setting main function visibility to public");
            module.walk([&](LLVM::LLVMFuncOp func) {
                if (func.getSymName() == "main") {
                    PGX_INFO("🎯 Found main function! Setting sym_visibility=\"public\" for ExecutionEngine");
                    func.setSymVisibilityAttr(mlir::StringAttr::get(&getContext(), "public"));
                    PGX_INFO("✅ Main function visibility set to public");
                } else {
                    PGX_INFO("📝 Function " + func.getSymName().str() + " kept with default visibility");
                }
            });
            PGX_INFO("🎉 Post-processing completed!");
        }
    }
};

} // namespace

std::unique_ptr<Pass> createStandardToLLVMPass() {
    PGX_INFO("Creating StandardToLLVMPass instance");
    return std::make_unique<StandardToLLVMPass>();
}

} // namespace pgx_lower
} // namespace mlir