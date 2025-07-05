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
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

using namespace mlir; // NOLINT(*-build-using-namespace)
using namespace mlir::pg; // NOLINT(*-build-using-namespace)

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.scan_table to the current low-level implementation
class ScanTableOpLowering final : public OpConversionPattern<ScanTableOp> {
   public:
    explicit ScanTableOpLowering(LLVMTypeConverter &typeConverter)
    : OpConversionPattern<ScanTableOp>(typeConverter, &typeConverter.getContext()) {}

    auto matchAndRewrite(ScanTableOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        const auto tableName = op.getTableName();

        // Create a proper string constant for the table name (instead of hashing it!)
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
        auto tableNameStr = tableName.str();
        
        // Find the module to create the global string constant at module level
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (!module) {
            return failure();
        }
        
        // Create a global string constant in the module (at module scope, not function scope)
        auto stringType = LLVM::LLVMArrayType::get(rewriter.getI8Type(), tableNameStr.length() + 1);
        auto stringLiteral = rewriter.getStringAttr(tableNameStr + '\0');
        auto globalName = "_table_name_" + std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));

        // Insert the global at the module level (before the current function)
        auto insertionGuard = OpBuilder::InsertionGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        
        auto global = rewriter.create<LLVM::GlobalOp>(
            loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal, 
            globalName, stringLiteral
        );

        // Restore insertion point to the current operation
        rewriter.setInsertionPoint(op);

        // Get pointer to the global string
        auto tableNamePtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, global.getName());

        // Call the runtime function to open the table: (ptr) -> ptr
        auto openTableFn = SymbolRefAttr::get(ctx, "open_postgres_table");
        auto operands = llvm::SmallVector<Value>{tableNamePtr};
        const auto ptrTableHandle = rewriter.create<func::CallOp>(loc, ptrType, openTableFn, operands).getResult(0);

        // The result is already the correct type (ptr), no need for conversion cast
        // The type converter will handle the type mapping automatically
        rewriter.replaceOp(op, ptrTableHandle);

        return success();
    }
};

/// Lower pg.read_tuple to runtime function call
class ReadTupleOpLowering final : public OpConversionPattern<ReadTupleOp> {
   public:
    explicit ReadTupleOpLowering(LLVMTypeConverter &typeConverter)
    : OpConversionPattern<ReadTupleOp>(typeConverter, &typeConverter.getContext()) {}

    auto matchAndRewrite(ReadTupleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        // Use the converted table handle (should be ptr now)
        auto tableHandle = adaptor.getTableHandle();

        // Call the runtime function: read_next_tuple_from_table(ptr) -> i64
        auto readTupleFn = SymbolRefAttr::get(ctx, "read_next_tuple_from_table");
        auto operands = llvm::SmallVector<Value>{tableHandle};
        const auto tupleI64 = rewriter.create<func::CallOp>(loc, rewriter.getI64Type(), readTupleFn, operands).getResult(0);

        // The result is i64, but the type converter expects ptr for tuple handles
        // Convert the i64 to ptr to match the expected LLVM representation
        auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
        auto tuplePtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, tupleI64);

        rewriter.replaceOp(op, tuplePtr);

        return success();
    }
};

class GetIntFieldOpLowering final : public OpConversionPattern<GetIntFieldOp> {
   public:
    explicit GetIntFieldOpLowering(LLVMTypeConverter &typeConverter)
    : OpConversionPattern<GetIntFieldOp>(typeConverter, &typeConverter.getContext()) {}

    auto matchAndRewrite(GetIntFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();

        // Use the converted tuple handle (should be ptr now)
        const auto tuple = adaptor.getTuple();
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

        // Call runtime function: get_int_field(ptr, i32, ptr) -> i32
        auto operands = llvm::SmallVector<Value>{tuple, fieldIndexVal, nullFlagPtr};
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

/// Convert string constant placeholder integers to LLVM string globals
class StringConstantLowering final : public OpRewritePattern<LLVM::IntToPtrOp> {
   public:
    explicit StringConstantLowering(MLIRContext *context) : OpRewritePattern(context) {}

    LogicalResult matchAndRewrite(LLVM::IntToPtrOp op, PatternRewriter &rewriter) const override {
        // Check if this is a string constant placeholder
        auto constantOp = op.getOperand().getDefiningOp<arith::ConstantOp>();
        if (!constantOp) {
            return failure(); // Not a constant
        }
        
        // Check if this constant has the pg.string_value attribute
        auto stringValueAttr = constantOp->getAttr("pg.string_value");
        if (!stringValueAttr) {
            return failure(); // Not a string constant placeholder
        }
        
        auto stringAttr = mlir::dyn_cast<StringAttr>(stringValueAttr);
        if (!stringAttr) {
            return failure(); // Invalid string attribute
        }
        
        std::string stringValue = stringAttr.getValue().str();
        auto loc = op.getLoc();
        
        // Get unique ID for naming the global
        auto intAttr = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        uint64_t placeholderId = intAttr ? intAttr.getValue().getZExtValue() : 0;
        
        // Create LLVM string global for this constant
        auto module = op->getParentOfType<ModuleOp>();
        if (!module) {
            return failure();
        }
        
        auto stringType = LLVM::LLVMArrayType::get(rewriter.getI8Type(), stringValue.length() + 1);
        auto stringLiteral = rewriter.getStringAttr(stringValue + '\0');
        auto globalName = "_string_const_" + std::to_string(placeholderId);
        
        // Check if global already exists
        auto existingGlobal = module.lookupSymbol<LLVM::GlobalOp>(globalName);
        if (!existingGlobal) {
            // Create the global at module level
            auto insertionGuard = OpBuilder::InsertionGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            
            existingGlobal = rewriter.create<LLVM::GlobalOp>(
                loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal,
                globalName, stringLiteral
            );
        }
        
        // Replace IntToPtrOp with AddressOfOp to the global
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto addressOfOp = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, existingGlobal.getName());
        rewriter.replaceOp(op, addressOfOp);
        
        return success();
    }
};

/// Clean up UnrealizedConversionCastOp from tuple handles to i64
class UnrealizedConversionCastOpLowering final : public OpConversionPattern<UnrealizedConversionCastOp> {
   public:
    explicit UnrealizedConversionCastOpLowering(LLVMTypeConverter &typeConverter)
    : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter, &typeConverter.getContext()) {}

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        // Handle all unrealized conversion casts that are blocking LLVM IR translation
        auto inputs = adaptor.getInputs();
        if (inputs.size() == 1 && op.getResults().size() == 1) {
            const auto input = inputs[0];
            const auto inputType = input.getType();
            const auto resultType = op.getResults()[0].getType();

            // Case 1: !llvm.ptr -> i64: Convert to llvm.ptrtoint
            if (mlir::isa<LLVM::LLVMPointerType>(inputType) && mlir::isa<IntegerType>(resultType)) {
                auto ptrToIntValue = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), resultType, input);
                rewriter.replaceOp(op, ptrToIntValue);
                return success();
            }
            
            // Case 2: i64 -> !llvm.ptr: Convert to llvm.inttoptr  
            if (mlir::isa<IntegerType>(inputType) && mlir::isa<LLVM::LLVMPointerType>(resultType)) {
                auto intToPtrValue = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), resultType, input);
                rewriter.replaceOp(op, intToPtrValue);
                return success();
            }
            
            // Case 3: i64 -> !pg.tuple_handle: This should be an IntToPtr conversion (for backwards compatibility)
            if (mlir::isa<IntegerType>(inputType) && mlir::isa<TupleHandleType>(resultType)) {
                auto ptrType = rewriter.getType<LLVM::LLVMPointerType>();
                auto ptrValue = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), ptrType, input);
                rewriter.replaceOp(op, ptrValue);
                return success();
            }
            
            // Case 4: !pg.tuple_handle -> !llvm.ptr: Direct replacement (both are pointers)
            if (mlir::isa<TupleHandleType>(inputType) && mlir::isa<LLVM::LLVMPointerType>(resultType)) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 5: !pg.table_handle -> !llvm.ptr: Direct replacement (both are pointers)
            if (mlir::isa<TableHandleType>(inputType) && mlir::isa<LLVM::LLVMPointerType>(resultType)) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 6: i64 -> i64 or any other direct type match: Direct replacement
            if (inputType == resultType) {
                rewriter.replaceOp(op, input);
                return success();
            }
            
            // Case 7: Any other conversion: Direct replacement for simplicity
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
class PgCmpOpLowering final : public OpConversionPattern<PgCmpOp> {
public:
    explicit PgCmpOpLowering(LLVMTypeConverter &typeConverter) : OpConversionPattern<PgCmpOp>(typeConverter, &typeConverter.getContext()), typeConverter(typeConverter) {}

    auto matchAndRewrite(PgCmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const -> LogicalResult override {
        const auto loc = op.getLoc();
        auto left = adaptor.getLeft();
        auto right = adaptor.getRight();
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

private:
    LLVMTypeConverter &typeConverter;
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
        
        // Simple bitwise AND for boolean values
        auto result = rewriter.create<arith::AndIOp>(loc, leftBool, rightBool);
        rewriter.replaceOp(op, result);
        
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
        
        // Simple bitwise OR for boolean values
        auto result = rewriter.create<arith::OrIOp>(loc, leftBool, rightBool);
        rewriter.replaceOp(op, result);
        
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

struct LowerPgToSCFPass final : OperationPass<mlir::ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPgToSCFPass)
    
    LowerPgToSCFPass() : OperationPass(TypeID::get<LowerPgToSCFPass>()) {}

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<pg::PgDialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        // Set up the type converter for pg types to LLVM types
        mlir::LLVMTypeConverter typeConverter(ctx);
        auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
        
        // Add type conversions for pg dialect types
        typeConverter.addConversion([=](mlir::pg::TableHandleType) {
            return ptrType; // !pg.table_handle -> !llvm.ptr
        });
        typeConverter.addConversion([=](mlir::pg::TupleHandleType) {
            return ptrType; // !pg.tuple_handle -> !llvm.ptr  
        });

        // Set up conversion target
        mlir::ConversionTarget target(*ctx);
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        
        // Mark pg dialect as illegal to force conversion
        target.addIllegalDialect<mlir::pg::PgDialect>();

        // Walk through all functions in the module
        module.walk([&](func::FuncOp func) {
            // Count pg operations before lowering
            bool hasPgOps = false;
            func.walk([&](mlir::Operation* op) {
                std::string opName = op->getName().getStringRef().str();
                if (opName.substr(0, 3) == "pg.") {
                    hasPgOps = true;
                }
            });

            if (!hasPgOps) {
                return; // No pg operations in this function
            }

            // Set up conversion patterns using ConversionPattern instead of OpRewritePattern
            mlir::RewritePatternSet patterns(ctx);
            patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering, PgCmpOpLowering, UnrealizedConversionCastOpLowering>(typeConverter);
            
            // Add OpRewritePattern-based lowering for operations that don't need type conversion
            patterns.add<PgAndOpLowering, PgOrOpLowering, PgNotOpLowering>(ctx);
            
            // Add arithmetic operation lowering patterns
            patterns.add<PgAddOpLowering, PgSubOpLowering, PgMulOpLowering, PgDivOpLowering, PgModOpLowering>(ctx);
            
            // Add text field access lowering pattern
            patterns.add<GetTextFieldOpLowering>(ctx);
            
            // Add string constant lowering pattern
            patterns.add<StringConstantLowering>(ctx);
            
            // Apply dialect conversion instead of greedy pattern application
            if (failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        });
    }

    [[nodiscard]] auto getName() const -> StringRef override { 
        return "lower-pg-to-scf"; 
    }
    
    [[nodiscard]] auto getArgument() const -> StringRef override { 
        return "lower-pg-to-scf"; 
    }
    
    [[nodiscard]] auto getDescription() const -> StringRef override {
        return "Lower PostgreSQL dialect operations to SCF and standard dialects";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<LowerPgToSCFPass>();
    }
};;;

} // namespace

void pg::populatePgToSCFConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
    auto &llvmTypeConverter = static_cast<LLVMTypeConverter&>(typeConverter);
    patterns.add<ScanTableOpLowering, ReadTupleOpLowering, GetIntFieldOpLowering>(llvmTypeConverter);
}

auto pg::createLowerPgToSCFPass() -> std::unique_ptr<OperationPass<mlir::ModuleOp>> {
    return std::make_unique<LowerPgToSCFPass>();
}