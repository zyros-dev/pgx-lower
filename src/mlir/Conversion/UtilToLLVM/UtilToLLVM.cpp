#include "mlir/Conversion/UtilToLLVM/UtilToLLVM.h"
#include "execution/logging.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace pgx::mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

static LLVM::LLVMStructType convertTuple(TupleType tupleType, const TypeConverter &typeConverter) {
    std::vector<Type> types;
    for (auto t : tupleType.getTypes()) {
        types.push_back(typeConverter.convertType(t));
    }
    return LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Pattern: util.alloc → palloc call
class AllocOpLowering : public OpConversionPattern<util::AllocOp> {
public:
    using OpConversionPattern<util::AllocOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(util::AllocOp allocOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = allocOp->getLoc();
        auto refType = allocOp.getType().cast<util::RefType>();
        auto elemType = refType.getElementType();
        
        MLIR_PGX_DEBUG("UtilToLLVM", "Lowering util.alloc for type");
        
        // Calculate size needed
        auto module = allocOp->getParentOfType<ModuleOp>();
        auto context = rewriter.getContext();
        auto i64Type = IntegerType::get(context, 64);
        auto i8Type = IntegerType::get(context, 8);
        auto i8PtrType = LLVM::LLVMPointerType::get(context);
        
        // Get size of the element type
        Value sizeBytes;
        auto llvmElemType = typeConverter->convertType(elemType);
        if (llvmElemType.isa<IntegerType>()) {
            // For integer types, calculate size based on bit width
            auto intType = llvmElemType.cast<IntegerType>();
            unsigned bitWidth = intType.getWidth();
            unsigned byteSize = (bitWidth + 7) / 8;
            sizeBytes = rewriter.create<LLVM::ConstantOp>(loc, i64Type, 
                                                           rewriter.getI64IntegerAttr(byteSize));
        } else if (llvmElemType.isa<LLVM::LLVMStructType>()) {
            // For struct types (tuples), we need to calculate the size
            // For now, use a simple heuristic - this should be improved
            // to use proper DataLayout calculation
            auto structType = llvmElemType.cast<LLVM::LLVMStructType>();
            unsigned totalSize = 0;
            for (auto fieldType : structType.getBody()) {
                if (auto intType = fieldType.dyn_cast<IntegerType>()) {
                    totalSize += (intType.getWidth() + 7) / 8;
                } else {
                    // Default to 8 bytes for unknown types
                    totalSize += 8;
                }
            }
            sizeBytes = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                                           rewriter.getI64IntegerAttr(totalSize));
        } else {
            // Default to pointer size (8 bytes)
            sizeBytes = rewriter.create<LLVM::ConstantOp>(loc, i64Type, 
                                                           rewriter.getI64IntegerAttr(8));
        }
        
        // Look up or create palloc function
        auto pallocFunc = LLVM::lookupOrCreateFn(module, "palloc", {i64Type}, i8PtrType);
        if (failed(pallocFunc)) {
            return failure();
        }
        
        // Call palloc
        auto allocatedPtr = rewriter.create<LLVM::CallOp>(
            loc, TypeRange{i8PtrType}, SymbolRefAttr::get(*pallocFunc), 
            ValueRange{sizeBytes}
        );
        Value rawPtr = allocatedPtr.getResult();
        
        // Cast to typed pointer
        auto typedPtrType = LLVM::LLVMPointerType::get(context);
        auto typedPtr = rewriter.create<LLVM::BitcastOp>(loc, typedPtrType, rawPtr);
        
        rewriter.replaceOp(allocOp, typedPtr);
        return success();
    }
};

// Pattern: util.pack → LLVM struct construction
class PackOpLowering : public OpConversionPattern<util::PackOp> {
public:
    using OpConversionPattern<util::PackOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(util::PackOp packOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto tupleType = packOp.getType().cast<TupleType>();
        auto structType = convertTuple(tupleType, *getTypeConverter());
        
        MLIR_PGX_DEBUG("UtilToLLVM", "Lowering util.pack to LLVM struct");
        
        // Create undefined struct
        Value result = rewriter.create<LLVM::UndefOp>(packOp->getLoc(), structType);
        
        // Insert each value
        unsigned pos = 0;
        for (auto val : adaptor.getValues()) {
            result = rewriter.create<LLVM::InsertValueOp>(packOp->getLoc(), result, val, pos++);
        }
        
        rewriter.replaceOp(packOp, result);
        return success();
    }
};

// Pattern: util.get_tuple → LLVM extractvalue
class GetTupleOpLowering : public OpConversionPattern<util::GetTupleOp> {
public:
    using OpConversionPattern<util::GetTupleOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(util::GetTupleOp getTupleOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto resType = typeConverter->convertType(getTupleOp.getType());
        
        auto index = getTupleOp.getIndex();
        MLIR_PGX_DEBUG("UtilToLLVM", "Lowering util.get_tuple");
        
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
            getTupleOp, resType, adaptor.getTuple(), index);
        
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct UtilToLLVMLoweringPass : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UtilToLLVMLoweringPass)
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, arith::ArithDialect>();
    }
    
    void runOnOperation() final {
        MLIR_PGX_INFO("UtilToLLVM", "Starting Util to LLVM lowering pass");
        
        auto module = getOperation();
        auto context = &getContext();
        
        // Set up type converter
        LLVMTypeConverter typeConverter(context);
        
        // Add conversions for our custom types
        typeConverter.addConversion([&](TupleType tupleType) {
            return convertTuple(tupleType, typeConverter);
        });
        
        typeConverter.addConversion([&](util::RefType refType) -> Type {
            MLIR_PGX_DEBUG("UtilToLLVM", "Converting util.ref type to LLVM pointer");
            auto context = refType.getContext();
            return LLVM::LLVMPointerType::get(context);
        });
        
        // Populate conversion patterns
        RewritePatternSet patterns(context);
        populateUtilToLLVMConversionPatterns(typeConverter, patterns);
        
        // Set up conversion target
        ConversionTarget target(*context);
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();
        target.addLegalOp<ModuleOp>();
        
        // Mark Util operations as illegal
        target.addIllegalDialect<util::UtilDialect>();
        
        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            MLIR_PGX_ERROR("UtilToLLVM", "Failed to lower Util operations to LLVM");
            signalPassFailure();
            return;
        }
        
        MLIR_PGX_INFO("UtilToLLVM", "Successfully completed Util to LLVM lowering");
    }
    
    StringRef getArgument() const final { return "util-to-llvm"; }
    StringRef getDescription() const final { 
        return "Lower Util dialect operations to LLVM dialect"; 
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void pgx::mlir::populateUtilToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                                      RewritePatternSet &patterns) {
    patterns.add<AllocOpLowering, PackOpLowering, GetTupleOpLowering>(
        typeConverter, patterns.getContext());
    
    MLIR_PGX_DEBUG("UtilToLLVM", "Registered Util to LLVM conversion patterns");
}

std::unique_ptr<Pass> pgx::mlir::createUtilToLLVMPass() {
    return std::make_unique<UtilToLLVMLoweringPass>();
}