//===----------------------------------------------------------------------===//
//
// Lowering pass from DB dialect to LLVM dialect
//
//===----------------------------------------------------------------------===//

#include "compiler/Conversion/DBToLLVM/LowerDBToLLVM.h"
#include "compiler/Dialect/DB/DBDialect.h"
#include "compiler/Dialect/DB/DBOps.h"
#include "execution/logging.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::db;

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

class DBTypeConverter : public LLVMTypeConverter {
public:
    DBTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
        // For now, simplify type conversion - just pass through most types
        // TODO Phase 9: Add proper nullable type conversion when needed
        
        // Standard LLVM type converter handles most built-in types
        // We'll add specific DB type conversions as needed
    }
};

//===----------------------------------------------------------------------===//
// Operation Lowering Patterns
//===----------------------------------------------------------------------===//

class ConstantOpLowering : public OpConversionPattern<pgx_lower::compiler::dialect::db::ConstantOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto attr = op.getValue();
        
        Value constant;
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            constant = rewriter.create<LLVM::ConstantOp>(loc, op.getType(), intAttr);
        } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
            constant = rewriter.create<LLVM::ConstantOp>(loc, op.getType(), floatAttr);
        } else {
            return failure();
        }
        
        rewriter.replaceOp(op, constant);
        return success();
    }
};

class AsNullableOpLowering : public OpConversionPattern<pgx_lower::compiler::dialect::db::AsNullableOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    
    LogicalResult matchAndRewrite(AsNullableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto structType = getTypeConverter()->convertType(op.getType());
        
        // Create struct {value, is_null=false}
        auto undef = rewriter.create<LLVM::UndefOp>(loc, structType);
        auto withValue = rewriter.create<LLVM::InsertValueOp>(
            loc, structType, undef, adaptor.getVal(), ArrayRef<int64_t>{0});
        auto falseVal = rewriter.create<LLVM::ConstantOp>(
            loc, IntegerType::get(op.getContext(), 1), 
            rewriter.getBoolAttr(false));
        auto result = rewriter.create<LLVM::InsertValueOp>(
            loc, structType, withValue, falseVal, ArrayRef<int64_t>{1});
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

class NullOpLowering : public OpConversionPattern<pgx_lower::compiler::dialect::db::NullOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    
    LogicalResult matchAndRewrite(NullOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto structType = getTypeConverter()->convertType(op.getType());
        
        // Create struct {value=undef, is_null=true}
        auto undef = rewriter.create<LLVM::UndefOp>(loc, structType);
        auto trueVal = rewriter.create<LLVM::ConstantOp>(
            loc, IntegerType::get(op.getContext(), 1), 
            rewriter.getBoolAttr(true));
        auto result = rewriter.create<LLVM::InsertValueOp>(
            loc, structType, undef, trueVal, ArrayRef<int64_t>{1});
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

template<typename BinaryOp>
class BinaryOpLowering : public OpConversionPattern<BinaryOp> {
public:
    using OpConversionPattern<BinaryOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // For now, just pass through the left operand value
        // Full NULL handling would check both operands for NULL
        rewriter.replaceOp(op, adaptor.getLeft());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerDBToLLVMPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDBToLLVMPass)
    
    LowerDBToLLVMPass() : OperationPass(TypeID::get<LowerDBToLLVMPass>()) {}
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, arith::ArithDialect, scf::SCFDialect>();
    }
    
    void runOnOperation() override {
        auto module = getOperation();
        auto &context = getContext();
        
        // Log the module before lowering
        PGX_DEBUG("=== DB → LLVM Lowering Pass Started ===");
        PGX_DEBUG("Module before DB → LLVM lowering:");
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) {
            std::string moduleStr;
            llvm::raw_string_ostream rso(moduleStr);
            module.print(rso);
            PGX_DEBUG(rso.str());
        }
        
        DBTypeConverter typeConverter(&context);
        ConversionTarget target(context);
        
        // DB dialect is illegal (but we're not generating any DB ops yet)
        target.addIllegalDialect<DBDialect>();
        
        // Everything else is legal
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect, 
                              func::FuncDialect, scf::SCFDialect>();
        target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
        
        // PG dialect remains legal since we're not lowering it yet
        target.markUnknownOpDynamicallyLegal([](Operation *op) {
            return true;
        });
        
        RewritePatternSet patterns(&context);
        // TODO Phase 5: Implement proper lowering patterns
        patterns.add<ConstantOpLowering>(typeConverter, &context);
        patterns.add<AsNullableOpLowering>(typeConverter, &context);
        patterns.add<NullOpLowering>(typeConverter, &context);
        patterns.add<BinaryOpLowering<pgx_lower::compiler::dialect::db::AddOp>>(typeConverter, &context);
        patterns.add<BinaryOpLowering<pgx_lower::compiler::dialect::db::SubOp>>(typeConverter, &context);
        patterns.add<BinaryOpLowering<pgx_lower::compiler::dialect::db::MulOp>>(typeConverter, &context);
        patterns.add<BinaryOpLowering<pgx_lower::compiler::dialect::db::DivOp>>(typeConverter, &context);
        
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
        
        // Log the module after lowering
        PGX_DEBUG("Module after DB → LLVM lowering:");
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) {
            std::string moduleStr;
            llvm::raw_string_ostream rso(moduleStr);
            module.print(rso);
            PGX_DEBUG(rso.str());
        }
        PGX_DEBUG("=== DB → LLVM Lowering Pass Completed ===");
    }
    
    StringRef getName() const override { return "LowerDBToLLVMPass"; }
    
    StringRef getArgument() const override { return "lower-db-to-llvm"; }
    
    StringRef getDescription() const override {
        return "Lower Database dialect to LLVM dialect";
    }
    
    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<LowerDBToLLVMPass>(*this);
    }
};

} // namespace

namespace pgx_lower { namespace compiler { namespace dialect { namespace db {

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createLowerDBToLLVMPass() {
    return std::make_unique<LowerDBToLLVMPass>();
}

}}}} // namespace pgx_lower::compiler::dialect::db