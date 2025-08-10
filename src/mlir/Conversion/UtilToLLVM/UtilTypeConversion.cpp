#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      assert(typeConverter->convertTypes(op->getResultTypes(), convertedTypes).succeeded());
      rewriter.replaceOpWithNewOp<Op>(op, convertedTypes, ValueRange(operands), op->getAttrs());
      return success();
   }
};

class SizeOfLowering :  public OpConversionPattern<pgx::mlir::util::SizeOfOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::SizeOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::SizeOfOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<pgx::mlir::util::SizeOfOp>(op, rewriter.getIndexType(), TypeAttr::get(typeConverter->convertType(op.getType())));
      return success();
   }
};

} // end anonymous namespace

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void pgx::mlir::util::populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::GetTupleOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::UndefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::PackOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::UnPackOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::ToGenericMemrefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::ToMemrefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::IsRefValidOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::InvalidRefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::StoreOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::AllocOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::AllocaOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::DeAllocOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::GenericMemrefCastOp>>(typeConverter, patterns.getContext());
   patterns.add<SizeOfLowering>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::LoadOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::TupleElementPtrOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::ArrayElementPtrOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<pgx::mlir::util::FilterTaggedPtr>>(typeConverter, patterns.getContext());

   typeConverter.addConversion([&](pgx::mlir::util::RefType genericMemrefType) {
      return pgx::mlir::util::RefType::get(genericMemrefType.getContext(), typeConverter.convertType(genericMemrefType.getElementType()));
   });
   typeConverter.addConversion([&](pgx::mlir::util::VarLen32Type varType) {
      return varType;
   });
}
