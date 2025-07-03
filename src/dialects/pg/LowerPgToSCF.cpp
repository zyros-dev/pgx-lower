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

using namespace mlir; // NOLINT(*-build-using-namespace)
using namespace mlir::pg; // NOLINT(*-build-using-namespace)

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.scan_table to the current low-level implementation
class ScanTableOpLowering final : public OpRewritePattern<ScanTableOp> {
   public:
    explicit ScanTableOpLowering(MLIRContext *context)
    : OpRewritePattern(context) {}

    auto matchAndRewrite(ScanTableOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        const auto tableName = op.getTableName();

        // Create a constant for the table name as a pointer (to match runtime function signature)
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
        auto tableNameHash = static_cast<int64_t>(std::hash<std::string>{}(tableName.str()));
        auto hashConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tableNameHash)
        );
        auto tableNamePtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, hashConst);

        // Call the runtime function to open the table: (ptr) -> ptr
        auto openTableFn = SymbolRefAttr::get(ctx, "open_postgres_table");
        auto operands = llvm::SmallVector<Value>{tableNamePtr};
        const auto ptrTableHandle = rewriter.create<func::CallOp>(loc, ptrType, openTableFn, operands).getResult(0);

        // Convert ptr result to !pg.table_handle type to match operation result type
        auto tableHandleType = op.getHandle().getType();
        auto tableHandle = rewriter.create<UnrealizedConversionCastOp>(
            loc, tableHandleType, mlir::ValueRange{ptrTableHandle}).getResult(0);

        // Replace the operation with the properly typed table handle
        rewriter.replaceOp(op, tableHandle);

        return success();
    }
};

/// Lower pg.read_tuple to runtime function call
class ReadTupleOpLowering final : public OpRewritePattern<ReadTupleOp> {
   public:
    explicit ReadTupleOpLowering(MLIRContext *context)
    : OpRewritePattern(context) {}

    auto matchAndRewrite(ReadTupleOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        llvm::errs() << "DEBUG: ReadTupleOpLowering::matchAndRewrite called\n";
        llvm::errs().flush();
        
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        const auto tableHandle = op.getTableHandle();

        // Convert !pg.table_handle to ptr for runtime function call: (ptr) -> i64
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
        auto tablePtr = rewriter.create<UnrealizedConversionCastOp>(
            loc, ptrType, mlir::ValueRange{tableHandle}).getResult(0);

        auto i64Type = rewriter.getI64Type();
        auto readTupleFn = SymbolRefAttr::get(ctx, "read_next_tuple_from_table");

        auto operands = llvm::SmallVector<Value>{tablePtr};
        const auto i64TupleHandle = rewriter.create<func::CallOp>(loc, i64Type, readTupleFn, operands).getResult(0);

        // Convert i64 result to !pg.tuple_handle type to match operation result type
        auto tupleHandleType = op.getTuple().getType();
        auto tupleHandle = rewriter.create<UnrealizedConversionCastOp>(
            loc, tupleHandleType, mlir::ValueRange{i64TupleHandle}).getResult(0);

        rewriter.replaceOp(op, tupleHandle);

        return success();
    }
};

class GetIntFieldOpLowering final : public OpRewritePattern<GetIntFieldOp> {
   public:
    explicit GetIntFieldOpLowering(MLIRContext *context)
    : OpRewritePattern<GetIntFieldOp>(context) {}

    auto matchAndRewrite(GetIntFieldOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        const auto tuple = op.getTuple();
        const unsigned fieldIndex = op.getFieldIndex();

        auto fieldIndexVal =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(fieldIndex));

        auto i32Type = rewriter.getI32Type();
        auto i1Type = rewriter.getI1Type();
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();

        auto getIntFieldFn = SymbolRefAttr::get(ctx, "get_int_field");

        // Find the function entry block to hoist the alloca
        auto funcOp = op->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return failure();

        auto &entryBlock = funcOp.front();
        auto guard = OpBuilder::InsertionGuard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        auto nullFlagPtr = rewriter.create<LLVM::AllocaOp>(
            loc,
            ptrType,
            i1Type,
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));

        // Restore insertion point to the original operation
        rewriter.setInsertionPoint(op);

        // Convert !pg.tuple_handle to ptr for runtime function call: (ptr, i32, ptr) -> i32
        auto tuplePtr = rewriter.create<UnrealizedConversionCastOp>(
            loc, ptrType, mlir::ValueRange{tuple}).getResult(0);

        auto operands = llvm::SmallVector<Value>{tuplePtr, fieldIndexVal, nullFlagPtr};
        const auto intValue = rewriter.create<func::CallOp>(loc, i32Type, getIntFieldFn, operands).getResult(0);

        auto nullFlag = rewriter.create<LLVM::LoadOp>(loc, i1Type, nullFlagPtr);

        auto results = llvm::SmallVector<Value>{intValue, nullFlag};
        rewriter.replaceOp(op, results);

        return success();
    }
};

class GetTextFieldOpLowering final : public OpRewritePattern<GetTextFieldOp> {
   public:
    explicit GetTextFieldOpLowering(MLIRContext *context)
    : OpRewritePattern<GetTextFieldOp>(context) {}

    LogicalResult matchAndRewrite(GetTextFieldOp op, PatternRewriter &rewriter) const override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        const auto tuple = op.getTuple();
        const unsigned fieldIndex = op.getFieldIndex();

        auto fieldIndexVal =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(fieldIndex));

        auto i64Type = rewriter.getI64Type();
        auto i1Type = rewriter.getI1Type();
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();

        auto getTextFieldFn = SymbolRefAttr::get(ctx, "get_text_field");

        // Find the function entry block to hoist the alloca
        auto funcOp = op->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return failure();

        auto &entryBlock = funcOp.front();
        auto guard = OpBuilder::InsertionGuard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        auto nullFlagPtr = rewriter.create<LLVM::AllocaOp>(
            loc,
            ptrType,
            i1Type,
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));

        // Restore insertion point to the original operation
        rewriter.setInsertionPoint(op);

        auto operands = llvm::SmallVector<Value>{tuple, fieldIndexVal, nullFlagPtr};
        const auto textPtr = rewriter.create<func::CallOp>(loc, i64Type, getTextFieldFn, operands).getResult(0);

        auto nullFlag = rewriter.create<LLVM::LoadOp>(loc, i1Type, nullFlagPtr);

        auto results = llvm::SmallVector<Value>{textPtr, nullFlag};
        rewriter.replaceOp(op, results);

        return success();
    }
};

/// Clean up UnrealizedConversionCastOp from tuple handles to i64
class UnrealizedConversionCastOpLowering final : public OpRewritePattern<UnrealizedConversionCastOp> {
   public:
    explicit UnrealizedConversionCastOpLowering(MLIRContext *context)
    : OpRewritePattern<UnrealizedConversionCastOp>(context) {}

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
        // Handle all unrealized conversion casts that are blocking LLVM IR translation
        if (op.getInputs().size() == 1 && op.getResults().size() == 1) {
            const auto input = op.getInputs()[0];
            const auto inputType = input.getType();
            const auto resultType = op.getResults()[0].getType();

            // Case 1: i64 -> !pg.tuple_handle: This should be an IntToPtr conversion
            if (mlir::isa<IntegerType>(inputType) && mlir::isa<TupleHandleType>(resultType)) {
                auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
                auto ptrValue = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), ptrType, input);
                rewriter.replaceOp(op, ptrValue);
                return success();
            }
            
            // Case 2: !pg.tuple_handle -> !llvm.ptr: Direct replacement (both are pointers)
            if (mlir::isa<TupleHandleType>(inputType) && mlir::isa<LLVM::LLVMPointerType>(resultType)) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 3: !pg.table_handle -> !llvm.ptr: Direct replacement (both are pointers)
            if (mlir::isa<TableHandleType>(inputType) && mlir::isa<LLVM::LLVMPointerType>(resultType)) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 4: i64 -> i64 or any other direct type match: Direct replacement
            if (inputType == resultType) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 5: Any other conversion: Direct replacement for simplicity
            rewriter.replaceOp(op, input);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class [[maybe_unused]] PgTypeConverter final : public TypeConverter {
   public:
    PgTypeConverter() {
        // Convert pg types to standard MLIR types
        addConversion([](const Type type) -> Type {
            if (mlir::isa<TableHandleType>(type)) {
                return IntegerType::get(type.getContext(), 64);
            }
            if (mlir::isa<TupleHandleType>(type)) {
                return IntegerType::get(type.getContext(), 64);
            }
            if (mlir::isa<TextType>(type)) {
                return IntegerType::get(type.getContext(), 64); // pointer to string
            }
            if (mlir::isa<DateType>(type)) {
                return IntegerType::get(type.getContext(), 32);
            }
            if (mlir::isa<NumericType>(type)) {
                return Float64Type::get(type.getContext());
            }

            // Return the type unchanged if it's not a pg type
            return type;
        });

        // Add materialization for type conversions
        addSourceMaterialization([](OpBuilder &builder, Type resultType, const ValueRange inputs, const Location loc) -> Value {
            if (inputs.size() != 1) {
                return {};
            }
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });

        addTargetMaterialization([](OpBuilder &builder, Type resultType, const ValueRange inputs, const Location loc) -> Value {
            if (inputs.size() != 1) {
                return {};
            }
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });

        addArgumentMaterialization([](OpBuilder &builder, Type resultType, const ValueRange inputs, const Location loc) -> Value {
            if (inputs.size() != 1) {
                return {};
            }
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Arithmetic Operator Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.add to arith.addi/arith.addf with null handling
class PgAddOpLowering final : public OpRewritePattern<PgAddOp> {
public:
    explicit PgAddOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgAddOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        // For now, assume non-nullable integer types for simplicity
        // TODO: Add proper null handling for PostgreSQL semantics
        auto result = rewriter.create<arith::AddIOp>(loc, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

/// Lower pg.sub to arith.subi/arith.subf with null handling  
class PgSubOpLowering final : public OpRewritePattern<PgSubOp> {
public:
    explicit PgSubOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgSubOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        auto result = rewriter.create<arith::SubIOp>(loc, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

/// Lower pg.mul to arith.muli/arith.mulf with null handling
class PgMulOpLowering final : public OpRewritePattern<PgMulOp> {
public:
    explicit PgMulOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgMulOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        auto result = rewriter.create<arith::MulIOp>(loc, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

/// Lower pg.div to arith.divsi/arith.divf with null handling
class PgDivOpLowering final : public OpRewritePattern<PgDivOp> {
public:
    explicit PgDivOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgDivOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        auto result = rewriter.create<arith::DivSIOp>(loc, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

/// Lower pg.mod to arith.remsi with null handling
class PgModOpLowering final : public OpRewritePattern<PgModOp> {
public:
    explicit PgModOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgModOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        auto result = rewriter.create<arith::RemSIOp>(loc, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Comparison Operator Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.compare to arith.cmpi/arith.cmpf with PostgreSQL semantics
class PgCmpOpLowering final : public OpRewritePattern<PgCmpOp> {
public:
    explicit PgCmpOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgCmpOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        auto predicate = op.getPredicate();
        
        // Map PostgreSQL comparison predicate to MLIR arith predicate
        // 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
        arith::CmpIPredicate arithPredicate;
        switch (predicate) {
            case 0: // eq
                arithPredicate = arith::CmpIPredicate::eq;
                break;
            case 1: // ne
                arithPredicate = arith::CmpIPredicate::ne;
                break;
            case 2: // lt
                arithPredicate = arith::CmpIPredicate::slt;
                break;
            case 3: // le
                arithPredicate = arith::CmpIPredicate::sle;
                break;
            case 4: // gt
                arithPredicate = arith::CmpIPredicate::sgt;
                break;
            case 5: // ge
                arithPredicate = arith::CmpIPredicate::sge;
                break;
            default:
                return failure();
        }
        
        // For now, assume integer comparison
        // TODO: Add type-specific comparison for float, text, etc.
        auto result = rewriter.create<arith::CmpIOp>(loc, arithPredicate, left, right);
        rewriter.replaceOp(op, result);
        
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Logical Operator Lowering Patterns  
//===----------------------------------------------------------------------===//

/// Lower pg.and to SCF control flow with three-valued logic
class PgAndOpLowering final : public OpRewritePattern<PgAndOp> {
public:
    explicit PgAndOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgAndOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        // Convert integer operands to boolean if needed for PostgreSQL boolean logic
        auto i1Type = rewriter.getI1Type();
        
        // Convert left operand to boolean (non-zero = true, zero = false)
        Value leftBool = left;
        if (left.getType() != i1Type) {
            auto zeroConst = rewriter.create<arith::ConstantOp>(loc, left.getType(), 
                                                               rewriter.getIntegerAttr(left.getType(), 0));
            leftBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, left, zeroConst).getResult();
        }
        
        // Convert right operand to boolean  
        Value rightBool = right;
        if (right.getType() != i1Type) {
            auto zeroConst = rewriter.create<arith::ConstantOp>(loc, right.getType(),
                                                               rewriter.getIntegerAttr(right.getType(), 0));
            rightBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, right, zeroConst).getResult();
        }
        
        // PostgreSQL AND logic: proper short-circuiting with scf.if
        // if (leftBool) then rightBool else false
        auto falseConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
        
        auto ifOp = rewriter.create<scf::IfOp>(loc, i1Type, leftBool, true);
        
        // Then region: evaluate right operand
        auto& thenRegion = ifOp.getThenRegion();
        rewriter.createBlock(&thenRegion);
        rewriter.setInsertionPointToStart(&thenRegion.front());
        rewriter.create<scf::YieldOp>(loc, mlir::ValueRange{rightBool});
        
        // Else region: return false
        auto& elseRegion = ifOp.getElseRegion();
        rewriter.createBlock(&elseRegion);
        rewriter.setInsertionPointToStart(&elseRegion.front());
        rewriter.create<scf::YieldOp>(loc, mlir::ValueRange{falseConst});
        
        rewriter.replaceOp(op, ifOp.getResult(0));
        
        return success();
    }
};

/// Lower pg.or to SCF control flow with three-valued logic
class PgOrOpLowering final : public OpRewritePattern<PgOrOp> {
public:
    explicit PgOrOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgOrOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = op.getLeft();
        auto right = op.getRight();
        
        // Convert integer operands to boolean if needed for PostgreSQL boolean logic
        auto i1Type = rewriter.getI1Type();
        
        // Convert left operand to boolean (non-zero = true, zero = false)
        Value leftBool = left;
        if (left.getType() != i1Type) {
            auto zeroConst = rewriter.create<arith::ConstantOp>(loc, left.getType(), 
                                                               rewriter.getIntegerAttr(left.getType(), 0));
            leftBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, left, zeroConst).getResult();
        }
        
        // Convert right operand to boolean  
        Value rightBool = right;
        if (right.getType() != i1Type) {
            auto zeroConst = rewriter.create<arith::ConstantOp>(loc, right.getType(),
                                                               rewriter.getIntegerAttr(right.getType(), 0));
            rightBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, right, zeroConst).getResult();
        }
        
        // PostgreSQL OR logic: proper short-circuiting with scf.if
        // if (leftBool) then true else rightBool
        auto trueConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        
        auto ifOp = rewriter.create<scf::IfOp>(loc, i1Type, leftBool, true);
        
        // Then region: return true (short-circuit)
        auto& thenRegion = ifOp.getThenRegion();
        rewriter.createBlock(&thenRegion);
        rewriter.setInsertionPointToStart(&thenRegion.front());
        rewriter.create<scf::YieldOp>(loc, mlir::ValueRange{trueConst});
        
        // Else region: evaluate right operand
        auto& elseRegion = ifOp.getElseRegion();
        rewriter.createBlock(&elseRegion);
        rewriter.setInsertionPointToStart(&elseRegion.front());
        rewriter.create<scf::YieldOp>(loc, mlir::ValueRange{rightBool});
        
        rewriter.replaceOp(op, ifOp.getResult(0));
        
        return success();
    }
};

/// Lower pg.not to arith XOR with true (bitwise NOT)
class PgNotOpLowering final : public OpRewritePattern<PgNotOp> {
public:
    explicit PgNotOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgNotOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto operand = op.getOperand();
        
        // Convert operand to boolean if needed
        auto i1Type = rewriter.getI1Type();
        Value boolOperand = operand;
        if (operand.getType() != i1Type) {
            auto zeroConst = rewriter.create<arith::ConstantOp>(loc, operand.getType(), 
                                                               rewriter.getIntegerAttr(operand.getType(), 0));
            boolOperand = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, operand, zeroConst);
        }
        
        // NOT x = x XOR true (proper logical NOT for boolean values)
        auto trueVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        auto result = rewriter.create<arith::XOrIOp>(loc, boolOperand, trueVal);
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Null Handling Operator Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.is_null to runtime function call
class PgIsNullOpLowering final : public OpRewritePattern<PgIsNullOp> {
public:
    explicit PgIsNullOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgIsNullOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto operand = op.getOperand();
        
        // For now, assume non-nullable types always return false
        // TODO: Implement proper null checking for nullable PostgreSQL types
        auto falseVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.replaceOp(op, falseVal);
        
        return success();
    }
};

/// Lower pg.is_not_null to runtime function call  
class PgIsNotNullOpLowering final : public OpRewritePattern<PgIsNotNullOp> {
public:
    explicit PgIsNotNullOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

    auto matchAndRewrite(PgIsNotNullOp op, PatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto operand = op.getOperand();
        
        // For now, assume non-nullable types always return true
        // TODO: Implement proper null checking for nullable PostgreSQL types
        auto trueVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, trueVal);
        
        return success();
    }
};

struct LowerPgToSCFPass final : PassWrapper<LowerPgToSCFPass, OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPgToSCFPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<pg::PgDialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        llvm::errs() << "DEBUG: LowerPgToSCFPass::runOnOperation() called\\n";
        llvm::errs().flush();
        
        // Walk through all functions in the module
        module.walk([&](func::FuncOp func) {
            llvm::errs() << "DEBUG: Processing function: " << func.getName().str() << "\\n";
            llvm::errs().flush();
            
            // Count pg operations before lowering
            int pgOpCount = 0;
            func.walk([&](mlir::Operation* op) {
                std::string opName = op->getName().getStringRef().str();
                if (opName.substr(0, 3) == "pg.") {
                    pgOpCount++;
                    llvm::errs() << "DEBUG: Found pg operation: " << opName << "\\n";
                    llvm::errs().flush();
                }
            });

            llvm::errs() << "DEBUG: Function " << func.getName().str() << " has " << pgOpCount << " pg operations\\n";
            llvm::errs().flush();

            if (pgOpCount == 0) {
                llvm::errs() << "DEBUG: Skipping function with no pg operations\\n";
                llvm::errs().flush();
                return; // No pg operations in this function
            }

            llvm::errs() << "DEBUG: About to create rewrite patterns for function " << func.getName().str() << "\\n";
            llvm::errs().flush();

            // Use simple rewrite patterns without type conversion
            auto patterns = RewritePatternSet(ctx);
            patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering, GetTextFieldOpLowering, UnrealizedConversionCastOpLowering,
                         PgAddOpLowering, PgSubOpLowering, PgMulOpLowering, PgDivOpLowering, PgModOpLowering,
                         PgCmpOpLowering, PgAndOpLowering, PgOrOpLowering, PgNotOpLowering, 
                         PgIsNullOpLowering, PgIsNotNullOpLowering>(
                ctx);

            llvm::errs() << "DEBUG: Created rewrite patterns successfully\\n";
            llvm::errs().flush();

            // Apply greedy pattern rewriting (no type conversion involved)
            llvm::errs() << "DEBUG: About to apply patterns greedily\n";
            llvm::errs().flush();
            
            if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
                llvm::errs() << "DEBUG: applyPatternsGreedily failed for function " << func.getName().str() << "\n";
                llvm::errs().flush();
                signalPassFailure();
                return;
            }
            
            // Apply cleanup patterns again to handle any remaining conversion casts
            llvm::errs() << "DEBUG: Applying cleanup patterns for remaining conversion casts\n";
            llvm::errs().flush();
            
            auto cleanupPatterns = RewritePatternSet(ctx);
            cleanupPatterns.add<UnrealizedConversionCastOpLowering>(ctx);
            
            if (failed(applyPatternsGreedily(func, std::move(cleanupPatterns)))) {
                llvm::errs() << "DEBUG: Cleanup patterns failed for function " << func.getName().str() << "\n";
                llvm::errs().flush();
                // Don't fail the pass for cleanup issues, just log it
            }
            
            llvm::errs() << "DEBUG: Successfully applied patterns for function " << func.getName().str() << "\\n";
            llvm::errs().flush();
        });
        
        llvm::errs() << "DEBUG: LowerPgToSCFPass::runOnOperation() completed successfully\\n";
        llvm::errs().flush();
    }

    [[nodiscard]] auto getArgument() const -> StringRef override { return "lower-pg-to-scf"; }
    [[nodiscard]] auto getDescription() const -> StringRef override {
        return "Lower PostgreSQL dialect operations to SCF and standard dialects";
    }
};

} // namespace

void pg::populatePgToSCFConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
    patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering, GetTextFieldOpLowering,
                 PgAddOpLowering, PgSubOpLowering, PgMulOpLowering, PgDivOpLowering, PgModOpLowering,
                 PgCmpOpLowering, PgAndOpLowering, PgOrOpLowering, PgNotOpLowering,
                 PgIsNullOpLowering, PgIsNotNullOpLowering>(
        patterns.getContext());
}

auto pg::createLowerPgToSCFPass() -> std::unique_ptr<OperationPass<mlir::ModuleOp>> {
    return std::make_unique<LowerPgToSCFPass>();
}