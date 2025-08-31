// Heavy MLIR template instantiation isolated to this compilation unit
#include "DSAToStdPatterns.h"

// All the heavy MLIR includes isolated here
#include "lingodb/mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "runtime-defs/DataSourceIteration.h"

using namespace mlir;

namespace {

// Generic pattern for simple type conversions
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

class ScanSourceLowering : public OpConversionPattern<mlir::dsa::ScanSource> {
   public:
   using OpConversionPattern<mlir::dsa::ScanSource>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::ScanSource op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Type> types;
      auto parentModule = op->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp = parentModule.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), 
            llvm::StringRef("rt_get_execution_context"), 
            rewriter.getFunctionType({}, {mlir::util::RefType::get(getContext(), rewriter.getI8Type())}),
            llvm::ArrayRef<mlir::NamedAttribute>{rewriter.getNamedAttr("visibility", rewriter.getStringAttr("private"))});
      }

      mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, mlir::ValueRange{}).getResult(0);
      mlir::Value description = rewriter.create<mlir::util::CreateConstVarLen>(op->getLoc(), mlir::util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());
      auto rawPtr = rt::DataSourceIteration::start(rewriter, op->getLoc())({executionContext, description})[0];
      rewriter.replaceOp(op, rawPtr);
      return success();
   }
};

} // anonymous namespace

namespace mlir {
namespace dsa {

void registerAllDSAToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns, ConversionTarget& target, MLIRContext* context) {
    // Function interface patterns
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    
    mlir::dsa::populateScalarToStdPatterns(typeConverter, patterns);
    
    mlir::dsa::populateDSAToStdPatterns(typeConverter, patterns);
    mlir::dsa::populateCollectionsToStdPatterns(typeConverter, patterns);
    
    // Utility patterns  
    mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, context);
    patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, context);
    patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CondSkipOp>>(typeConverter, context);
    patterns.insert<ScanSourceLowering>(typeConverter, context);
}

} // namespace dsa
} // namespace mlir