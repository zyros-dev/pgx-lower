//===- LowerPgToSubOp.cpp - PG to SubOperator lowering -----------*- C++ -*-===//
//
// Lowering pass from PG dialect to SubOperator dialect
//
//===----------------------------------------------------------------------===//

#include "dialects/pg/LowerPgToSubOp.h"
#include "dialects/pg/PgDialect.h"
#include "dialects/subop/SubOpDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::pg;
using namespace mlir::subop;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PgToSubOpTypeConverter : public TypeConverter {
public:
    PgToSubOpTypeConverter() {
        // Convert pg.table_handle to subop.table
        addConversion([](Type type) -> Type {
            if (auto tableHandle = dyn_cast<TableHandleType>(type)) {
                // For now, create a simple tuple type for the table
                // In a real implementation, we'd derive this from table metadata
                auto ctx = type.getContext();
                SmallVector<Type> columnTypes;
                // Example: assume table has int and text columns
                columnTypes.push_back(IntegerType::get(ctx, 32));
                columnTypes.push_back(IntegerType::get(ctx, 64)); // text as pointer
                auto tupleType = TupleType::get(ctx, columnTypes);
                return TableType::get(ctx, tupleType);
            }
            // Keep other types unchanged
            return type;
        });
        
        // Add target materialization
        addTargetMaterialization([](OpBuilder &builder, Type resultType, 
                                   ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return {};
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
                         .getResult(0);
        });
    }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert pg.scan_table to subop.scan
class ScanTableToSubOpLowering : public OpConversionPattern<ScanTableOp> {
public:
    using OpConversionPattern<ScanTableOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ScanTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, keep pg.scan_table as is - it will be handled by the direct lowering
        // The full SubOp implementation would create proper streaming operations
        return failure();
    }
};

/// Convert pg.read_tuple to operations on subop.tuplestream
class ReadTupleToSubOpLowering : public OpConversionPattern<ReadTupleOp> {
public:
    using OpConversionPattern<ReadTupleOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ReadTupleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, keep pg.read_tuple as is - it will be handled by the direct lowering
        return failure();
    }
};

/// Convert pg.get_int_field to operations within subop regions
class GetIntFieldToSubOpLowering : public OpConversionPattern<GetIntFieldOp> {
public:
    using OpConversionPattern<GetIntFieldOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(GetIntFieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, keep pg.get_int_field as is - it will be handled by the direct lowering
        return failure();
    }
};

/// Convert pg arithmetic/comparison operations to be used within SubOp regions
template<typename PgOp, typename ArithOp>
class PgArithToSubOpLowering : public OpRewritePattern<PgOp> {
public:
    using OpRewritePattern<PgOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(PgOp op, PatternRewriter &rewriter) const override {
        // These operations are typically used within subop.map or subop.filter regions
        // For now, lower them to standard arithmetic
        rewriter.replaceOpWithNewOp<ArithOp>(op, op.getLeft(), op.getRight());
        return success();
    }
};

/// Convert pg.cmp to arith.cmpi
class PgCmpToSubOpLowering : public OpRewritePattern<PgCmpOp> {
public:
    using OpRewritePattern<PgCmpOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(PgCmpOp op, PatternRewriter &rewriter) const override {
        // Map pg comparison predicates to arith predicates
        arith::CmpIPredicate arithPred;
        switch (op.getPredicate()) {
            case 0: arithPred = arith::CmpIPredicate::eq; break;
            case 1: arithPred = arith::CmpIPredicate::ne; break;
            case 2: arithPred = arith::CmpIPredicate::slt; break;
            case 3: arithPred = arith::CmpIPredicate::sle; break;
            case 4: arithPred = arith::CmpIPredicate::sgt; break;
            case 5: arithPred = arith::CmpIPredicate::sge; break;
            default: return failure();
        }
        
        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            op, arithPred, op.getLeft(), op.getRight());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerPgToSubOpPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPgToSubOpPass)
    
    LowerPgToSubOpPass() : OperationPass(TypeID::get<LowerPgToSubOpPass>()) {}
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<subop::SubOpDialect, arith::ArithDialect, func::FuncDialect>();
    }
    
    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();
        
        PgToSubOpTypeConverter typeConverter;
        
        // Set up conversion target
        ConversionTarget target(*ctx);
        // For now, mark PG dialect as legal since we're not actually lowering it
        // The real implementation will make it illegal and provide proper conversions
        target.addLegalDialect<pg::PgDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation *op) {
            return true;
        });
        
        // Set up conversion patterns
        RewritePatternSet patterns(ctx);
        patterns.add<ScanTableToSubOpLowering, ReadTupleToSubOpLowering, 
                    GetIntFieldToSubOpLowering>(typeConverter, ctx);
        
        // Add arithmetic lowering patterns
        patterns.add<PgArithToSubOpLowering<PgAddOp, arith::AddIOp>,
                    PgArithToSubOpLowering<PgSubOp, arith::SubIOp>,
                    PgArithToSubOpLowering<PgMulOp, arith::MulIOp>,
                    PgArithToSubOpLowering<PgDivOp, arith::DivSIOp>,
                    PgArithToSubOpLowering<PgModOp, arith::RemSIOp>,
                    PgCmpToSubOpLowering>(ctx);
        
        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getName() const override { 
        return "lower-pg-to-subop"; 
    }
    
    StringRef getArgument() const override { 
        return "lower-pg-to-subop"; 
    }
    
    StringRef getDescription() const override {
        return "Lower PostgreSQL dialect to SubOperator dialect";
    }
    
    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<LowerPgToSubOpPass>(*this);
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void pg::populatePgToSubOpConversionPatterns(RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
    patterns.add<ScanTableToSubOpLowering, ReadTupleToSubOpLowering,
                GetIntFieldToSubOpLowering>(typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> pg::createLowerPgToSubOpPass() {
    return std::make_unique<LowerPgToSubOpPass>();
}