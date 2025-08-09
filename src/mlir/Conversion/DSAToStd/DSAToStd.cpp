//===- DSAToStd.cpp - DSA to Standard dialects conversion pass -----------===//

#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "execution/logging.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

namespace mlir {

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class DSAToStdTypeConverter : public TypeConverter {
public:
    DSAToStdTypeConverter(MLIRContext *context) {
        // DSA TableBuilder → opaque pointer (util.ref<i8>)
        addConversion([context](pgx::mlir::dsa::TableBuilderType type) -> Type {
            MLIR_PGX_DEBUG("DSAToStd", "Converting TableBuilder type to util.ref<i8>");
            auto i8Type = IntegerType::get(context, 8);
            return pgx::mlir::util::RefType::get(context, i8Type);
        });
        
        // Keep standard types as-is
        addConversion([](Type type) { return type; });
        
        // Handle tuple types (needed for table_builder<tuple<...>>)
        addConversion([](TupleType type) { return type; });
        
        // Add argument materialization for function boundaries
        addArgumentMaterialization([context](OpBuilder &builder, Type resultType,
                                     ValueRange inputs, Location loc) -> Value {
            // When converting function arguments from util.ref<i8> to DSA types
            if (resultType.isa<pgx::mlir::dsa::TableBuilderType>() && !inputs.empty()) {
                auto inputType = inputs[0].getType();
                if (inputType.isa<pgx::mlir::util::RefType>()) {
                    MLIR_PGX_DEBUG("DSAToStd", "Materializing argument conversion from util.ref to DSA type");
                    return inputs[0]; // Direct use since it's already converted
                }
            }
            return Value();
        });
        
        // Add source materialization (util.ref<i8> → DSA types)
        addSourceMaterialization([context](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
            // This handles the case where we need to convert back for uses expecting DSA types
            if (resultType.isa<pgx::mlir::dsa::TableBuilderType>() && !inputs.empty()) {
                auto inputType = inputs[0].getType();
                auto i8Type = IntegerType::get(context, 8);
                auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                if (inputType == refType) {
                    MLIR_PGX_DEBUG("DSAToStd", "Materializing source conversion from util.ref to DSA type");
                    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
                }
            }
            return Value();
        });
        
        // Add target materialization (DSA types → util.ref<i8>)
        addTargetMaterialization([context](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
            auto i8Type = IntegerType::get(context, 8);
            auto expectedRefType = pgx::mlir::util::RefType::get(context, i8Type);
            
            if (resultType == expectedRefType && !inputs.empty()) {
                auto inputType = inputs[0].getType();
                if (inputType.isa<pgx::mlir::dsa::TableBuilderType>()) {
                    MLIR_PGX_DEBUG("DSAToStd", "Materializing target conversion from DSA type to util.ref");
                    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
                }
            }
            return Value();
        });
    }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static func::FuncOp getOrCreateRuntimeFunction(
    OpBuilder &builder,
    ModuleOp module,
    Location loc,
    StringRef name,
    ArrayRef<Type> argTypes,
    ArrayRef<Type> resultTypes) {
    
    auto funcOp = module.lookupSymbol<func::FuncOp>(name);
    if (!funcOp) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto funcType = builder.getFunctionType(argTypes, resultTypes);
        funcOp = builder.create<func::FuncOp>(loc, name, funcType);
        funcOp.setPrivate();
    }
    return funcOp;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//


// Pattern for dsa.create_ds → runtime call
class CreateDSToStdPattern : public OpConversionPattern<pgx::mlir::dsa::CreateDSOp> {
public:
    using OpConversionPattern<pgx::mlir::dsa::CreateDSOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::CreateDSOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        auto loc = op.getLoc();
        auto resultType = op.getResult().getType();
        
        // Only handle TableBuilder for now (Test 1 requirement)
        if (!resultType.isa<pgx::mlir::dsa::TableBuilderType>()) {
            MLIR_PGX_DEBUG("DSAToStd", "CreateDSOp: not a TableBuilder, skipping");
            return failure();
        }
        
        MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.create_ds for TableBuilder");
        
        // Get the schema attribute
        auto schemaAttr = op.getInitAttr();
        if (!schemaAttr) {
            MLIR_PGX_ERROR("DSAToStd", "TableBuilder create_ds missing schema attribute");
            return failure();
        }
        
        // Create global string for schema
        auto module = op->getParentOfType<ModuleOp>();
        auto globalName = (Twine("table_schema_") + Twine(reinterpret_cast<uintptr_t>(op.getOperation()))).str();
        auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
        if (!globalOp) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            
            auto strType = LLVM::LLVMArrayType::get(
                IntegerType::get(rewriter.getContext(), 8),
                schemaAttr->size() + 1);
            globalOp = rewriter.create<LLVM::GlobalOp>(
                loc, strType, true, LLVM::Linkage::Private,
                globalName, rewriter.getStringAttr(schemaAttr->str() + '\0'));
        }
        
        // Get pointer to the schema string
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto schemaPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, globalOp.getName());
        
        // Call runtime function to create table builder
        auto i8Type = IntegerType::get(rewriter.getContext(), 8);
        auto refType = pgx::mlir::util::RefType::get(rewriter.getContext(), i8Type);
        
        auto createFunc = getOrCreateRuntimeFunction(rewriter, module, loc,
            "pgx_runtime_create_table_builder", 
            {ptrType}, {refType});
        
        auto callOp = rewriter.create<func::CallOp>(loc, createFunc.getName(), TypeRange{refType}, ValueRange{schemaPtr});
        auto builderHandle = callOp.getResult(0);
        
        // Replace the operation
        rewriter.replaceOp(op, builderHandle);
        
        MLIR_PGX_DEBUG("DSAToStd", "Successfully converted dsa.create_ds");
        return success();
    }
};

// Pattern for dsa.ds_append → runtime calls
class DSAppendToStdPattern : public OpConversionPattern<pgx::mlir::dsa::DSAppendOp> {
public:
    using OpConversionPattern<pgx::mlir::dsa::DSAppendOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::DSAppendOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        auto loc = op.getLoc();
        auto dsType = op.getDs().getType();
        
        // Only handle TableBuilder for now
        if (!dsType.isa<pgx::mlir::dsa::TableBuilderType>()) {
            MLIR_PGX_DEBUG("DSAToStd", "DSAppendOp: not a TableBuilder, skipping");
            return failure();
        }
        
        MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.ds_append for TableBuilder");
        
        // Get the converted builder handle (should be util.ref<i8>)
        auto builderHandle = adaptor.getDs();
        if (!builderHandle) {
            MLIR_PGX_ERROR("DSAToStd", "Null builder handle in adaptor");
            return failure();
        }
        
        // The handle might still be DSA type if it's a function argument
        // The type converter will handle the conversion through materialization
        
        // Get the value to append from adaptor since it might be converted too
        auto value = adaptor.getVal();
        auto valueType = value.getType();
        
        // Ensure the builder handle is properly typed for runtime calls
        Value convertedHandle = builderHandle;
        if (!builderHandle.getType().isa<pgx::mlir::util::RefType>()) {
            // Need to convert the handle
            auto i8Type = IntegerType::get(rewriter.getContext(), 8);
            auto refType = pgx::mlir::util::RefType::get(rewriter.getContext(), i8Type);
            convertedHandle = rewriter.create<mlir::UnrealizedConversionCastOp>(
                loc, refType, builderHandle).getResult(0);
        }
        
        // Create function declarations for runtime append calls
        auto module = op->getParentOfType<ModuleOp>();
        auto i8Type = IntegerType::get(rewriter.getContext(), 8);
        auto refType = pgx::mlir::util::RefType::get(rewriter.getContext(), i8Type);
        
        // Handle nullable types
        if (valueType.isa<pgx::db::NullableI64Type>()) {
            MLIR_PGX_DEBUG("DSAToStd", "Handling nullable value in ds_append");
            
            // Extract null flag and raw value
            auto isNull = rewriter.create<pgx::db::IsNullOp>(loc, rewriter.getI1Type(), value);
            auto rawValue = rewriter.create<pgx::db::NullableGetValOp>(loc, rewriter.getI64Type(), value);
            
            // Call the nullable append function
            auto appendFunc = getOrCreateRuntimeFunction(rewriter, module, loc,
                "pgx_runtime_append_nullable_i64", 
                {refType, rewriter.getI1Type(), rewriter.getI64Type()}, {});
            
            
            rewriter.create<func::CallOp>(loc, appendFunc.getName(), TypeRange(), 
                ValueRange({convertedHandle, isNull, rawValue}));
            
        } else if (valueType.isInteger(64)) {
            // Direct value append (non-nullable)
            auto appendFunc = getOrCreateRuntimeFunction(rewriter, module, loc,
                "pgx_runtime_append_i64_direct", 
                {refType, rewriter.getI64Type()}, {});
            rewriter.create<func::CallOp>(loc, appendFunc.getName(), TypeRange(), 
                ValueRange({convertedHandle, value}));
        } else {
            MLIR_PGX_ERROR("DSAToStd", "Unsupported value type for ds_append");
            return failure();
        }
        
        // Erase the original operation
        rewriter.eraseOp(op);
        
        MLIR_PGX_DEBUG("DSAToStd", "Successfully converted dsa.ds_append");
        return success();
    }
};

// Pattern for dsa.next_row → runtime call
class NextRowToStdPattern : public OpConversionPattern<pgx::mlir::dsa::NextRowOp> {
public:
    using OpConversionPattern<pgx::mlir::dsa::NextRowOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::NextRowOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        auto loc = op.getLoc();
        
        MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.next_row");
        
        // Get the builder handle (should be util.ref<i8>)
        auto builderHandle = adaptor.getBuilder();
        if (!builderHandle) {
            MLIR_PGX_ERROR("DSAToStd", "Null builder handle in adaptor");
            return failure();
        }
        
        // The handle might still be DSA type if it's a function argument
        // The type converter will handle the conversion through materialization
        
        // Ensure the builder handle is properly typed for runtime calls
        Value convertedHandle = builderHandle;
        if (!builderHandle.getType().isa<pgx::mlir::util::RefType>()) {
            // Need to convert the handle
            auto i8Type = IntegerType::get(rewriter.getContext(), 8);
            auto refType = pgx::mlir::util::RefType::get(rewriter.getContext(), i8Type);
            convertedHandle = rewriter.create<mlir::UnrealizedConversionCastOp>(
                loc, refType, builderHandle).getResult(0);
        }
        
        // Create function declaration for runtime next row
        auto module = op->getParentOfType<ModuleOp>();
        auto i8Type = IntegerType::get(rewriter.getContext(), 8);
        auto refType = pgx::mlir::util::RefType::get(rewriter.getContext(), i8Type);
        
        auto nextRowFunc = getOrCreateRuntimeFunction(rewriter, module, loc,
            "pgx_runtime_table_next_row", {refType}, {});
        
        // Call runtime function with the converted handle
        rewriter.create<func::CallOp>(loc, nextRowFunc.getName(), TypeRange(), 
                                      ValueRange{convertedHandle});
        
        // Erase the original operation
        rewriter.eraseOp(op);
        
        MLIR_PGX_DEBUG("DSAToStd", "Successfully converted dsa.next_row");
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct DSAToStdPass : public PassWrapper<DSAToStdPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DSAToStdPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<pgx::mlir::dsa::DSADialect, pgx::db::DBDialect, pgx::mlir::util::UtilDialect,
                       func::FuncDialect, arith::ArithDialect, 
                       scf::SCFDialect, index::IndexDialect, LLVM::LLVMDialect>();
    }

    StringRef getArgument() const final { return "convert-dsa-to-std"; }
    StringRef getDescription() const final { 
        return "Convert DSA dialect operations to Standard MLIR with runtime calls"; 
    }

    void runOnOperation() override {
        MLIR_PGX_INFO("DSAToStd", "Starting DSA to Standard conversion pass");
        
        auto module = getOperation();
        auto *context = &getContext();
        
        // Debug: Print module before conversion
        MLIR_PGX_DEBUG("DSAToStd", "Module before conversion");

        // Set up type converter
        DSAToStdTypeConverter typeConverter(context);

        // Set up conversion target
        ConversionTarget target(*context);
        
        // DSA dialect is illegal (we're converting away from it)
        target.addIllegalDialect<pgx::mlir::dsa::DSADialect>();
        
        // Legal dialects
        target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                              scf::SCFDialect, index::IndexDialect, 
                              LLVM::LLVMDialect, pgx::db::DBDialect, 
                              pgx::mlir::util::UtilDialect, func::FuncDialect>();
        
        // Function operations need special handling for type conversion
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return typeConverter.isLegal(op.getOperandTypes());
        });
        target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
            return typeConverter.isLegal(op.getOperandTypes()) &&
                   typeConverter.isLegal(op.getResultTypes());
        });
        
        // Allow unrealized conversion casts temporarily during conversion
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        
        // Collect conversion patterns
        RewritePatternSet patterns(context);
        
        // Add our DSA conversion patterns
        patterns.add<CreateDSToStdPattern, DSAppendToStdPattern, NextRowToStdPattern>(
            typeConverter, context);
        
        // Add function signature conversion
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        
        // Add call operation conversion pattern
        populateCallOpTypeConversionPattern(patterns, typeConverter);
        
        // Add return op conversion pattern
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        
        // Apply the conversion
        // Using partial conversion to allow gradual type conversion through materialization
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            MLIR_PGX_ERROR("DSAToStd", "Failed to convert DSA operations to Standard MLIR");
            signalPassFailure();
            return;
        }
        
        // Debug: Print module after conversion
        MLIR_PGX_DEBUG("DSAToStd", "Module after conversion");
        
        
        MLIR_PGX_INFO("DSAToStd", "Successfully completed DSA to Standard conversion");
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createDSAToStdPass() {
    return std::make_unique<DSAToStdPass>();
}

} // namespace mlir