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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::pg;
using namespace pgx_lower::compiler::dialect::subop;

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
                // TODO: Create proper StateMembersAttr for PostgreSQL tables
                // For now, just return the original type to get compilation working
                return type;
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

/// Convert pg.scan_table to subop.generate pattern
class ScanTableToSubOpLowering : public OpConversionPattern<ScanTableOp> {
public:
    using OpConversionPattern<ScanTableOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ScanTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = op.getContext();
        
        // For now, create a placeholder that will be properly handled later
        // The actual generate pattern needs to be created at a higher level
        // where we have access to the full loop structure
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), ValueRange{});
        
        return success();
    }
};

/// Convert pg.read_tuple to operations on subop.tuplestream
class ReadTupleToSubOpLowering : public OpConversionPattern<ReadTupleOp> {
public:
    using OpConversionPattern<ReadTupleOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(ReadTupleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // pg.read_tuple doesn't have a direct equivalent in SubOp
        // The iteration is handled by SubOp's stream processing model
        // For now, replace with unrealized cast to pass through
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), adaptor.getTableHandle());
        return success();
    }
};

/// Convert pg.get_int_field to operations within subop regions
class GetIntFieldToSubOpLowering : public OpConversionPattern<GetIntFieldOp> {
public:
    using OpConversionPattern<GetIntFieldOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(GetIntFieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, replace with unrealized cast to pass through
        // This will be properly handled when we implement the full SubOp model
        SmallVector<Type> resultTypes = {op.getValue().getType(), op.getIsNull().getType()};
        SmallVector<Value> results;
        for (auto type : resultTypes) {
            auto cast = rewriter.create<UnrealizedConversionCastOp>(
                op.getLoc(), type, adaptor.getTuple());
            results.push_back(cast.getResult(0));
        }
        rewriter.replaceOp(op, results);
        return success();
    }
};

/// Convert pg.get_field (polymorphic) to operations within subop regions
class GetFieldToSubOpLowering : public OpConversionPattern<GetFieldOp> {
public:
    using OpConversionPattern<GetFieldOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(GetFieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        // For now, replace with unrealized cast to pass through
        // This will be properly handled when we implement the full SubOp model
        SmallVector<Type> resultTypes = {op.getValue().getType(), op.getIsNull().getType()};
        SmallVector<Value> results;
        for (auto type : resultTypes) {
            auto cast = rewriter.create<UnrealizedConversionCastOp>(
                op.getLoc(), type, adaptor.getTuple());
            results.push_back(cast.getResult(0));
        }
        rewriter.replaceOp(op, results);
        return success();
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
        // registry.insert<subop::SubOpDialect, arith::ArithDialect, func::FuncDialect,
        //                scf::SCFDialect, LLVM::LLVMDialect>();
        registry.insert<arith::ArithDialect, func::FuncDialect,
                       scf::SCFDialect, LLVM::LLVMDialect>();
    }
    
    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();
        
        // Log the module before lowering
        llvm::errs() << "=== PG → SubOp Lowering Pass Started ===\n";
        llvm::errs() << "Module before PG → SubOp lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n";
        
        PgToSubOpTypeConverter typeConverter;
        
        // Set up conversion target
        ConversionTarget target(*ctx);
        
        // Mark PG dialect as ILLEGAL - it must be lowered!
        target.addIllegalDialect<pg::PgDialect>();
        
        // Mark target dialects as legal
        // target.addLegalDialect<subop::SubOpDialect, arith::ArithDialect, 
        //                       func::FuncDialect>();
        target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
        // Also mark LLVM and SCF as legal since they might be present
        target.addLegalDialect<scf::SCFDialect, LLVM::LLVMDialect>();
        
        // Allow unrealized conversion casts for progressive lowering
        target.addLegalOp<UnrealizedConversionCastOp>();
        
        // Standard operations that don't need conversion
        target.addLegalOp<ModuleOp>();
        
        // Set up conversion patterns
        RewritePatternSet patterns(ctx);
        patterns.add<ScanTableToSubOpLowering, ReadTupleToSubOpLowering, 
                    GetIntFieldToSubOpLowering, GetFieldToSubOpLowering>(typeConverter, ctx);
        
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
        
        // Log the module after lowering
        llvm::errs() << "Module after PG → SubOp lowering:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n=== PG → SubOp Lowering Pass Completed ===\n\n";
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