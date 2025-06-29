#include "dialects/pg/LowerPgToSCF.h"
#include "dialects/pg/PgDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::pg;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.scan_table to the current low-level implementation
class ScanTableOpLowering : public OpRewritePattern<ScanTableOp> {
   public:
    explicit ScanTableOpLowering(MLIRContext *context)
    : OpRewritePattern<ScanTableOp>(context) {}

    LogicalResult matchAndRewrite(ScanTableOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        // Get the table name
        StringRef tableName = op.getTableName();

        // Create a constant for the table name as an integer (simplified for now)
        // In the real implementation, this would be a proper table lookup
        auto tableNameHash = static_cast<int64_t>(std::hash<std::string>{}(tableName.str()));
        Value tableNameConst =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tableNameHash));

        // Call the runtime function to open the table
        // This corresponds to the current @open_postgres_table call
        auto i64Type = rewriter.getI64Type();
        FlatSymbolRefAttr openTableFn = SymbolRefAttr::get(ctx, "open_postgres_table");

        // Create the function call
        llvm::SmallVector<Value> operands = {tableNameConst};
        Value tableHandle = rewriter.create<func::CallOp>(loc, i64Type, openTableFn, operands).getResult(0);

        // Replace the operation with the table handle (as i64)
        rewriter.replaceOp(op, tableHandle);

        return success();
    }
};

/// Lower pg.read_tuple to runtime function call
class ReadTupleOpLowering : public OpRewritePattern<ReadTupleOp> {
   public:
    explicit ReadTupleOpLowering(MLIRContext *context)
    : OpRewritePattern<ReadTupleOp>(context) {}

    LogicalResult matchAndRewrite(ReadTupleOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        Value tableHandle = op.getTableHandle();

        auto i64Type = rewriter.getI64Type();
        FlatSymbolRefAttr readTupleFn = SymbolRefAttr::get(ctx, "read_next_tuple_from_table");

        llvm::SmallVector<Value> operands = {tableHandle};
        Value tupleHandle = rewriter.create<func::CallOp>(loc, i64Type, readTupleFn, operands).getResult(0);

        rewriter.replaceOp(op, tupleHandle);

        return success();
    }
};

class GetIntFieldOpLowering : public OpRewritePattern<GetIntFieldOp> {
   public:
    explicit GetIntFieldOpLowering(MLIRContext *context)
    : OpRewritePattern<GetIntFieldOp>(context) {}

    LogicalResult matchAndRewrite(GetIntFieldOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        Value tuple = op.getTuple();
        unsigned fieldIndex = op.getFieldIndex();

        Value fieldIndexVal =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(fieldIndex));

        auto i32Type = rewriter.getI32Type();
        auto i1Type = rewriter.getI1Type();
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();

        FlatSymbolRefAttr getIntFieldFn = SymbolRefAttr::get(ctx, "get_int_field");

        // Find the function entry block to hoist the alloca
        auto funcOp = op->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return failure();

        auto &entryBlock = funcOp.front();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        Value nullFlagPtr = rewriter.create<LLVM::AllocaOp>(
            loc,
            ptrType,
            i1Type,
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));

        // Restore insertion point to the original operation
        rewriter.setInsertionPoint(op);

        llvm::SmallVector<Value> operands = {tuple, fieldIndexVal, nullFlagPtr};
        Value intValue = rewriter.create<func::CallOp>(loc, i32Type, getIntFieldFn, operands).getResult(0);

        Value nullFlag = rewriter.create<LLVM::LoadOp>(loc, i1Type, nullFlagPtr);

        llvm::SmallVector<Value> results = {intValue, nullFlag};
        rewriter.replaceOp(op, results);

        return success();
    }
};

class GetTextFieldOpLowering : public OpRewritePattern<GetTextFieldOp> {
   public:
    explicit GetTextFieldOpLowering(MLIRContext *context)
    : OpRewritePattern<GetTextFieldOp>(context) {}

    LogicalResult matchAndRewrite(GetTextFieldOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        Value tuple = op.getTuple();
        unsigned fieldIndex = op.getFieldIndex();

        Value fieldIndexVal =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(fieldIndex));

        auto i64Type = rewriter.getI64Type();
        auto i1Type = rewriter.getI1Type();
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();

        FlatSymbolRefAttr getTextFieldFn = SymbolRefAttr::get(ctx, "get_text_field");

        // Find the function entry block to hoist the alloca
        auto funcOp = op->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return failure();

        auto &entryBlock = funcOp.front();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        Value nullFlagPtr = rewriter.create<LLVM::AllocaOp>(
            loc,
            ptrType,
            i1Type,
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));

        // Restore insertion point to the original operation
        rewriter.setInsertionPoint(op);

        llvm::SmallVector<Value> operands = {tuple, fieldIndexVal, nullFlagPtr};
        Value textPtr = rewriter.create<func::CallOp>(loc, i64Type, getTextFieldFn, operands).getResult(0);

        Value nullFlag = rewriter.create<LLVM::LoadOp>(loc, i1Type, nullFlagPtr);

        llvm::SmallVector<Value> results = {textPtr, nullFlag};
        rewriter.replaceOp(op, results);

        return success();
    }
};

/// Clean up UnrealizedConversionCastOp from tuple handles to i64
class UnrealizedConversionCastOpLowering : public OpRewritePattern<UnrealizedConversionCastOp> {
   public:
    explicit UnrealizedConversionCastOpLowering(MLIRContext *context)
    : OpRewritePattern<UnrealizedConversionCastOp>(context) {}

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
        // If this is a cast from tuple handle to i64, just remove it since
        // the lowering pass already converts tuple handles to i64 values
        if (op.getInputs().size() == 1 && op.getResults().size() == 1) {
            Value input = op.getInputs()[0];
            Value result = op.getResults()[0];

            // If input is i64 and result is i64, this is a no-op
            if (mlir::isa<IntegerType>(input.getType()) && mlir::isa<IntegerType>(result.getType())) {
                rewriter.replaceOp(op, input);
                return success();
            }
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PgTypeConverter : public TypeConverter {
   public:
    PgTypeConverter() {
        // Convert pg types to standard MLIR types
        addConversion([](Type type) -> Type {
            if (mlir::isa<TableHandleType>(type))
                return IntegerType::get(type.getContext(), 64);
            if (mlir::isa<TupleHandleType>(type))
                return IntegerType::get(type.getContext(), 64);
            if (mlir::isa<TextType>(type))
                return IntegerType::get(type.getContext(), 64); // pointer to string
            if (mlir::isa<DateType>(type))
                return IntegerType::get(type.getContext(), 32);
            if (mlir::isa<NumericType>(type))
                return Float64Type::get(type.getContext());

            // Return the type unchanged if it's not a pg type
            return type;
        });

        // Add materializations for type conversions
        addSourceMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return {};
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });

        addTargetMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return {};
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });

        addArgumentMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return {};
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerPgToSCFPass : public PassWrapper<LowerPgToSCFPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPgToSCFPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, scf::SCFDialect, func::FuncDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        auto func = getOperation();
        MLIRContext *ctx = &getContext();

        // Use simple rewrite patterns without type conversion
        RewritePatternSet patterns(ctx);
        patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering, GetTextFieldOpLowering, UnrealizedConversionCastOpLowering>(
            ctx);

        // Apply greedy pattern rewriting (no type conversion involved)
        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
            signalPassFailure();
        }
    }

    StringRef getArgument() const override { return "lower-pg-to-scf"; }
    StringRef getDescription() const override {
        return "Lower PostgreSQL dialect operations to SCF and standard dialects";
    }
};

} // namespace

void mlir::pg::populatePgToSCFConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
    patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering, GetTextFieldOpLowering>(
        patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::pg::createLowerPgToSCFPass() {
    return std::make_unique<LowerPgToSCFPass>();
}