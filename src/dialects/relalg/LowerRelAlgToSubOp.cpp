//===- LowerRelAlgToSubOp.cpp - RelAlg to SubOp lowering -------*- C++ -*-===//
//
// Lowering pass from RelAlg dialect to SubOperator dialect
// Based on LingoDB's implementation
//
//===----------------------------------------------------------------------===//

#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class RelAlgToSubOpTypeConverter : public TypeConverter {
public:
    RelAlgToSubOpTypeConverter() {
        // Convert relalg.tuplestream to subop.tuplestream (they're the same)
        addConversion([](Type type) -> Type {
            if (auto tupleStream = dyn_cast<pgx_lower::compiler::dialect::tuples::TupleStreamType>(type)) {
                return type; // Keep as-is
            }
            // Keep other types unchanged
            return type;
        });
        
        // Add target materialization for progressive lowering
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

/// Convert relalg.basetable to subop operations
class BaseTableToSubOpLowering : public OpConversionPattern<relalg::BaseTableOp> {
public:
    using OpConversionPattern<relalg::BaseTableOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(relalg::BaseTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // For now, create a placeholder that represents the table scan
        // In a full implementation, this would:
        // 1. Create subop.get_external to get the table reference
        // 2. Create subop.scan to iterate over the table
        // 3. Set up column mappings
        
        // For now, just pass through with unrealized cast
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), ValueRange{});
        
        return success();
    }
};

/// Convert relalg.selection to subop.filter
class SelectionToSubOpLowering : public OpConversionPattern<relalg::SelectionOp> {
public:
    using OpConversionPattern<relalg::SelectionOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(relalg::SelectionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // relalg.selection becomes subop.filter
        // The predicate region is converted to produce a boolean
        
        // For now, just pass through
        rewriter.replaceOp(op, adaptor.getRel());
        return success();
    }
};

/// Convert relalg.map to subop operations
class MapToSubOpLowering : public OpConversionPattern<relalg::MapOp> {
public:
    using OpConversionPattern<relalg::MapOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(relalg::MapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // relalg.map computes new columns
        // This becomes a combination of subop operations to compute values
        
        // For now, just pass through
        rewriter.replaceOp(op, adaptor.getRel());
        return success();
    }
};

/// Convert relalg.aggregation to subop operations
class AggregationToSubOpLowering : public OpConversionPattern<relalg::AggregationOp> {
public:
    using OpConversionPattern<relalg::AggregationOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(relalg::AggregationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // relalg.aggregation becomes:
        // 1. subop.lookup to build hash table for groups
        // 2. subop.gather to get group values
        // 3. subop.scan to produce results
        
        // For now, just pass through
        rewriter.replaceOp(op, adaptor.getRel());
        return success();
    }
};

/// Convert relalg.join to subop operations
class InnerJoinToSubOpLowering : public OpConversionPattern<relalg::InnerJoinOp> {
public:
    using OpConversionPattern<relalg::InnerJoinOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(relalg::InnerJoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // Inner join becomes:
        // 1. subop.lookup to build hash table from right side
        // 2. subop.filter on left side to probe hash table
        
        // For now, just pass through left side
        rewriter.replaceOp(op, adaptor.getLeft());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

class LowerRelAlgToSubOpPass : public OperationPass<ModuleOp> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRelAlgToSubOpPass)
    
    LowerRelAlgToSubOpPass() : OperationPass(TypeID::get<LowerRelAlgToSubOpPass>()) {}
    
    StringRef getArgument() const final { return "lower-relalg-to-subop"; }
    StringRef getDescription() const final { return "Lower RelAlg dialect to SubOp dialect"; }
    StringRef getName() const final { return "LowerRelAlgToSubOpPass"; }
    
    std::unique_ptr<Pass> clonePass() const final {
        return std::make_unique<LowerRelAlgToSubOpPass>(*this);
    }
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<subop::SubOperatorDialect, 
                       arith::ArithDialect, 
                       func::FuncDialect,
                       scf::SCFDialect>();
    }
    
    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();
        
        // Ensure required dialects are loaded
        ctx->getOrLoadDialect<subop::SubOperatorDialect>();
        ctx->getOrLoadDialect<arith::ArithDialect>();
        ctx->getOrLoadDialect<func::FuncDialect>();
        ctx->getOrLoadDialect<scf::SCFDialect>();
        
        llvm::errs() << "=== RelAlg → SubOp Lowering Pass Started ===\n";
        llvm::errs() << "Module before lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n";
        
        RelAlgToSubOpTypeConverter typeConverter;
        
        // Set up conversion target
        ConversionTarget target(*ctx);
        
        // Mark RelAlg dialect as illegal - it must be lowered
        target.addIllegalDialect<relalg::RelAlgDialect>();
        
        // Mark target dialects as legal
        target.addLegalDialect<subop::SubOperatorDialect, 
                              arith::ArithDialect, 
                              func::FuncDialect,
                              scf::SCFDialect>();
        
        // Allow unrealized conversion casts for progressive lowering
        target.addLegalOp<UnrealizedConversionCastOp>();
        
        // Standard operations that don't need conversion
        target.addLegalOp<ModuleOp>();
        
        // Set up conversion patterns
        RewritePatternSet patterns(ctx);
        patterns.add<BaseTableToSubOpLowering, 
                    SelectionToSubOpLowering,
                    MapToSubOpLowering,
                    AggregationToSubOpLowering,
                    InnerJoinToSubOpLowering>(typeConverter, ctx);
        
        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        
        llvm::errs() << "Module after RelAlg → SubOp lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n=== RelAlg → SubOp Lowering Pass Completed ===\n";
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void pgx_lower::compiler::dialect::relalg::populateRelAlgToSubOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
    patterns.add<BaseTableToSubOpLowering, 
                SelectionToSubOpLowering,
                MapToSubOpLowering,
                AggregationToSubOpLowering,
                InnerJoinToSubOpLowering>(typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> 
pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass() {
    return std::make_unique<LowerRelAlgToSubOpPass>();
}