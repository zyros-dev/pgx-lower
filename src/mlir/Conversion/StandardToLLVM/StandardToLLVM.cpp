#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        PGX_DEBUG("StandardToLLVMPass: Starting unified Standard→LLVM conversion");
        
        auto* context = &getContext();
        ModuleOp module = getOperation();
        
        // Get DataLayoutAnalysis (like LingoDB)
        const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();
        PGX_DEBUG("StandardToLLVMPass: Retrieved DataLayoutAnalysis");
        
        // Create LLVM type converter with DataLayout options
        LowerToLLVMOptions options(context, dataLayoutAnalysis.getAtOrAbove(module));
        LLVMTypeConverter typeConverter(context, options, &dataLayoutAnalysis);
        PGX_DEBUG("StandardToLLVMPass: Created LLVM type converter with DataLayout");
        
        // Add source materialization (like LingoDB)
        typeConverter.addSourceMaterialization([&](OpBuilder&, FunctionType type, 
                                                  ValueRange valueRange, Location loc) {
            return valueRange.front();
        });
        
        // Create pattern set
        RewritePatternSet patterns(context);
        
        // Add ALL conversion patterns in the correct order
        PGX_DEBUG("StandardToLLVMPass: Populating SCF→ControlFlow patterns");
        populateSCFToControlFlowConversionPatterns(patterns);
        
        PGX_DEBUG("StandardToLLVMPass: Populating Func→LLVM patterns");
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        
        // CRITICAL: Add Util→LLVM patterns that were missing
        PGX_DEBUG("StandardToLLVMPass: Populating Util→LLVM patterns");
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
        
        PGX_DEBUG("StandardToLLVMPass: Populating Arith→LLVM patterns");
        arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        
        PGX_DEBUG("StandardToLLVMPass: Populating ControlFlow→LLVM patterns");
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        
        // Configure conversion target (use LLVMConversionTarget like LingoDB)
        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        
        PGX_DEBUG("StandardToLLVMPass: Configured LLVM conversion target");
        
        // Apply full conversion
        PGX_DEBUG("StandardToLLVMPass: Applying full conversion to module");
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("StandardToLLVMPass: Full conversion failed");
            signalPassFailure();
            return;
        }
        
        PGX_DEBUG("StandardToLLVMPass: Successfully completed unified Standard→LLVM conversion");
    }
};

} // namespace

std::unique_ptr<Pass> createStandardToLLVMPass() {
    PGX_DEBUG("Creating StandardToLLVMPass instance");
    return std::make_unique<StandardToLLVMPass>();
}

} // namespace pgx_lower
} // namespace mlir