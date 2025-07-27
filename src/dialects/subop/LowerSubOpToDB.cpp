//===- LowerSubOpToDB.cpp - SubOperator to DB lowering -----------*- C++ -*-===//
//
// Lowering pass from SubOperator dialect to Database dialect
//
//===----------------------------------------------------------------------===//

#include "dialects/subop/LowerSubOpToDB.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/db/DBDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::subop;
using namespace mlir::db;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class SubOpToDBTypeConverter : public TypeConverter {
public:
    SubOpToDBTypeConverter() {
        // Convert SubOp types to DB types
        addConversion([](Type type) -> Type {
            // Most types pass through unchanged
            return type;
        });
        
        // Convert tuple stream elements to nullable types when appropriate
        addConversion([](TupleStreamType streamType) -> Type {
            // Stream operations are lowered to loops, not a direct type conversion
            return streamType;
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

/// Convert subop.scan to SCF loop with DB operations
class ScanOpLowering : public OpConversionPattern<ScanOp> {
public:
    using OpConversionPattern<ScanOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // SubOp scan is typically lowered to a loop that iterates over the table
        // For now, we'll create a placeholder
        // In a real implementation, this would create an SCF while loop
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), adaptor.getTable());
        return success();
    }
};

/// Convert subop.filter to conditional DB operations
class FilterOpLowering : public OpConversionPattern<FilterOp> {
public:
    using OpConversionPattern<FilterOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(FilterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Filter operations in DB dialect would use conditional logic
        // The predicate region would be inlined with DB operations for NULL handling
        
        // For now, create a simple pass-through
        // Real implementation would create SCF if/then with DB is_null checks
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

/// Convert subop.map to DB operations
class MapOpLowering : public OpConversionPattern<MapOp> {
public:
    using OpConversionPattern<MapOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(MapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Map operations transform to DB operations with proper NULL handling
        // The mapper region would use DB arithmetic/logical operations
        
        // For now, create a pass-through
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

/// Convert subop.generate to SCF loop with DB operations
class GenerateOpLowering : public OpConversionPattern<GenerateOp> {
public:
    using OpConversionPattern<GenerateOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(GenerateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, we don't lower generate operations
        // The full implementation would create proper SCF loops with DB operations
        return failure();
    }
};

/// Convert subop.materialize to memory operations
class MaterializeOpLowering : public OpConversionPattern<MaterializeOp> {
public:
    using OpConversionPattern<MaterializeOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(MaterializeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // Materialize would allocate memory and store tuples
        // For now, pass through
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), adaptor.getStream());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerSubOpToDBPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSubOpToDBPass)
    
    LowerSubOpToDBPass() : OperationPass(TypeID::get<LowerSubOpToDBPass>()) {}
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<db::DBDialect, arith::ArithDialect, scf::SCFDialect,
                       func::FuncDialect, LLVM::LLVMDialect>();
    }
    
    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();
        
        // Log the module before lowering
        llvm::errs() << "=== SubOp → DB Lowering Pass Started ===\n";
        llvm::errs() << "Module before SubOp → DB lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n";
        
        SubOpToDBTypeConverter typeConverter;
        
        // Set up conversion target
        ConversionTarget target(*ctx);
        // Mark SubOp dialect as ILLEGAL - it must be lowered!
        target.addIllegalDialect<subop::SubOpDialect>();
        // Mark target dialects as legal
        target.addLegalDialect<db::DBDialect, arith::ArithDialect, 
                              scf::SCFDialect, func::FuncDialect,
                              LLVM::LLVMDialect>();
        // Allow unrealized conversion casts for progressive lowering
        target.addLegalOp<UnrealizedConversionCastOp>();
        // Standard operations that don't need conversion
        target.addLegalOp<ModuleOp>();
        
        // Set up conversion patterns
        RewritePatternSet patterns(ctx);
        patterns.add<ScanOpLowering, FilterOpLowering, MapOpLowering,
                    GenerateOpLowering, MaterializeOpLowering>(typeConverter, ctx);
        
        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
        
        // Log the module after lowering
        llvm::errs() << "Module after SubOp → DB lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n=== SubOp → DB Lowering Pass Completed ===\n\n";
    }
    
    StringRef getName() const override { 
        return "lower-subop-to-db"; 
    }
    
    StringRef getArgument() const override { 
        return "lower-subop-to-db"; 
    }
    
    StringRef getDescription() const override {
        return "Lower SubOperator dialect to Database dialect";
    }
    
    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<LowerSubOpToDBPass>(*this);
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void subop::populateSubOpToDBConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
    patterns.add<ScanOpLowering, FilterOpLowering, MapOpLowering,
                GenerateOpLowering, MaterializeOpLowering>(
                    typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> subop::createLowerSubOpToDBPass() {
    return std::make_unique<LowerSubOpToDBPass>();
}