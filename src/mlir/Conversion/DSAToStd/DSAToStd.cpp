//===- DSAToStd.cpp - DSA to Standard Lowering Pass ----------------------===//
//
// This file implements the lowering from DSA (Data Structure Abstraction) 
// dialect to Standard dialects using the DialectConversion framework.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "execution/logging.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Runtime function helpers (following LingoDB pattern)
//===----------------------------------------------------------------------===//

// Helper to get or create runtime function declarations
static func::FuncOp getOrCreateRuntimeFunction(OpBuilder& builder, ModuleOp module, 
                                               Location loc, StringRef name, 
                                               TypeRange argTypes, TypeRange resultTypes) {
    auto funcOp = module.lookupSymbol<func::FuncOp>(name);
    if (!funcOp) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        funcOp = builder.create<func::FuncOp>(loc, name, 
            builder.getFunctionType(argTypes, resultTypes));
        funcOp.setPrivate();
    }
    return funcOp;
}

//===----------------------------------------------------------------------===//
// CreateTableBuilder Lowering Pattern (following LingoDB)
//===----------------------------------------------------------------------===//

class CreateTableBuilderLowering : public OpConversionPattern<pgx::mlir::dsa::CreateDS> {
public:
    using OpConversionPattern<pgx::mlir::dsa::CreateDS>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(pgx::mlir::dsa::CreateDS createOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        // Only handle TableBuilder type
        if (!createOp.getDs().getType().isa<pgx::mlir::dsa::TableBuilderType>()) {
            return failure();
        }
        
        PGX_DEBUG("Converting dsa.create_ds for TableBuilder");
        
        auto loc = createOp.getLoc();
        auto* context = rewriter.getContext();
        
        // Get the schema attribute
        auto schemaAttr = createOp.getInitAttr();
        if (!schemaAttr) {
            return rewriter.notifyMatchFailure(createOp, "missing schema attribute");
        }
        
        // Create a global string for the schema
        auto parentModule = createOp->getParentOfType<ModuleOp>();
        auto schemaStr = schemaAttr.value().cast<StringAttr>().getValue();
        
        // Create global string constant
        auto globalName = (Twine("table_schema_") + 
                          Twine(reinterpret_cast<uintptr_t>(createOp.getOperation()))).str();
        
        LLVM::GlobalOp globalOp;
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            
            auto strType = LLVM::LLVMArrayType::get(
                IntegerType::get(context, 8), schemaStr.size() + 1);
            
            globalOp = rewriter.create<LLVM::GlobalOp>(
                loc, strType, true, LLVM::Linkage::Private,
                globalName, rewriter.getStringAttr(schemaStr.str() + '\0'));
        }
        
        // Get pointer to the schema string
        auto ptrType = LLVM::LLVMPointerType::get(context);
        auto schemaPtr = rewriter.create<LLVM::AddressOfOp>(
            loc, ptrType, globalOp.getName());
        
        // Create runtime call to create table builder
        auto i8Type = IntegerType::get(context, 8);
        auto refType = pgx::mlir::util::RefType::get(context, i8Type);
        
        auto createFunc = getOrCreateRuntimeFunction(
            rewriter, parentModule, loc,
            "pgx_runtime_create_table_builder",
            TypeRange{ptrType}, TypeRange{refType});
        
        auto callOp = rewriter.create<func::CallOp>(
            loc, createFunc, ValueRange{schemaPtr});
        
        // Replace the operation
        rewriter.replaceOp(createOp, callOp.getResult(0));
        
        PGX_DEBUG("Successfully converted dsa.create_ds to runtime call");
        return success();
    }
};

//===----------------------------------------------------------------------===//
// DSA to Standard Conversion Pass
//===----------------------------------------------------------------------===//

struct DSAToStdPass : public PassWrapper<DSAToStdPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DSAToStdPass)
    
    StringRef getArgument() const override { return "dsa-to-std"; }
    StringRef getDescription() const override { 
        return "Lower DSA dialect to Standard dialects"; 
    }
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<pgx::mlir::dsa::DSADialect>();
        registry.insert<pgx::mlir::util::UtilDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<LLVM::LLVMDialect>();
    }
    
    void runOnOperation() override {
        PGX_INFO("Starting DSAToStd conversion pass");
        
        auto module = getOperation();
        auto* context = &getContext();
        
        // Set up type converter
        TypeConverter typeConverter;
        
        // Default conversion - types convert to themselves
        typeConverter.addConversion([](Type type) { return type; });
        
        // DSA type conversions
        typeConverter.addConversion([context](pgx::mlir::dsa::TableBuilderType type) -> Type {
            // TableBuilder converts to util.ref<i8>
            return pgx::mlir::util::RefType::get(context, IntegerType::get(context, 8));
        });
        
        // Helper to check if a type range contains DSA types
        auto hasDSAType = [&](TypeRange types) {
            return llvm::any_of(types, [&](Type t) {
                auto converted = typeConverter.convertType(t);
                return converted && converted != t;
            });
        };
        
        // Set up conversion target
        ConversionTarget target(*context);
        
        // DSA operations are illegal
        target.addIllegalDialect<pgx::mlir::dsa::DSADialect>();
        
        // These dialects are legal
        target.addLegalDialect<pgx::mlir::util::UtilDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        
        // Func dialect operations are legal only if they don't use DSA types
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return !hasDSAType(op.getFunctionType().getInputs()) &&
                   !hasDSAType(op.getFunctionType().getResults());
        });
        target.addDynamicallyLegalOp<func::CallOp>([&](Operation* op) {
            return !hasDSAType(op->getOperandTypes()) && !hasDSAType(op->getResultTypes());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation* op) {
            return !hasDSAType(op->getOperandTypes());
        });
        
        // Unrealized conversion casts are temporarily allowed
        target.addLegalOp<UnrealizedConversionCastOp>();
        
        // Populate conversion patterns
        RewritePatternSet patterns(context);
        patterns.add<CreateTableBuilderLowering>(typeConverter, context);
        
        // Add patterns to convert function signatures
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        populateCallOpTypeConversionPattern(patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        
        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("DSAToStd conversion failed");
            signalPassFailure();
            return;
        }
        
        PGX_INFO("DSAToStd conversion completed successfully");
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createDSAToStdPass() {
    return std::make_unique<DSAToStdPass>();
}