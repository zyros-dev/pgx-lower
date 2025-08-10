#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Dialect/SCF/SCF.h>


using namespace mlir;
namespace {

class CreateFlagLowering : public OpConversionPattern<pgx::mlir::dsa::CreateFlag> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::CreateFlag>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::CreateFlag op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto boolType = rewriter.getI1Type();
      Type memrefType = util::RefType::get(rewriter.getContext(), boolType);
      Value alloca;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         auto func = op->getParentOfType<mlir::func::FuncOp>();
         rewriter.setInsertionPointToStart(&func.getBody().front());
         alloca = rewriter.create<pgx::mlir::util::AllocaOp>(op->getLoc(), memrefType, Value());
      }
      Value falseVal = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), boolType, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<util::StoreOp>(op->getLoc(), falseVal, alloca, Value());
      rewriter.replaceOp(op, alloca);
      return success();
   }
};
class SetFlagLowering : public OpConversionPattern<pgx::mlir::dsa::SetFlag> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::SetFlag>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::SetFlag op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.create<util::StoreOp>(op->getLoc(), adaptor.val(), adaptor.flag(), Value());
      rewriter.eraseOp(op);
      return success();
   }
};
class GetFlagLowering : public OpConversionPattern<pgx::mlir::dsa::GetFlag> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::GetFlag>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::GetFlag op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value flagValue = rewriter.create<util::LoadOp>(op->getLoc(), rewriter.getI1Type(), adaptor.flag(), Value());
      rewriter.replaceOp(op, flagValue);
      return success();
   }
};

} // namespace
void pgx::mlir::dsa::populateScalarToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](pgx::mlir::dsa::FlagType type) {
      return util::RefType::get(patterns.getContext(), IntegerType::get(type.getContext(), 1));
   });
   patterns.insert<CreateFlagLowering, SetFlagLowering, GetFlagLowering>(typeConverter, patterns.getContext());
}