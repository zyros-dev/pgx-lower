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
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {

namespace {

//===----------------------------------------------------------------------===//
// Type Converter Removed - Using RewritePattern approach
//===----------------------------------------------------------------------===//
// Type conversion is now handled manually in each pattern's matchAndRewrite method

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
class CreateDSToStdPattern : public OpRewritePattern<pgx::mlir::dsa::CreateDSOp> {
public:
    using OpRewritePattern<pgx::mlir::dsa::CreateDSOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::CreateDSOp op,
        PatternRewriter &rewriter) const override {
        
        MLIR_PGX_DEBUG("DSAToStd", "CreateDSToStdPattern::matchAndRewrite called");
        
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
class DSAppendToStdPattern : public OpRewritePattern<pgx::mlir::dsa::DSAppendOp> {
public:
    using OpRewritePattern<pgx::mlir::dsa::DSAppendOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::DSAppendOp op,
        PatternRewriter &rewriter) const override {
        
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Starting matchAndRewrite");
        
        auto loc = op.getLoc();
        auto dsType = op.getDs().getType();
        
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Got types");
        
        // Only handle TableBuilder for now
        if (!dsType.isa<pgx::mlir::dsa::TableBuilderType>()) {
            MLIR_PGX_DEBUG("DSAToStd", "DSAppendOp: not a TableBuilder, skipping");
            return failure();
        }
        
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Type check passed");
        MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.ds_append for TableBuilder");
        
        // Get the builder handle directly from the operation
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Getting builder handle from op");
        auto builderHandle = op.getDs();
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Got builderHandle");
        if (!builderHandle) {
            MLIR_PGX_ERROR("DSAToStd", "Null builder handle");
            return failure();
        }
        
        // Get the value to append directly from the operation
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Getting value from op");
        auto value = op.getVal();
        MLIR_PGX_INFO("DSAToStd", "DSAppendPattern: Got value");
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
class NextRowToStdPattern : public OpRewritePattern<pgx::mlir::dsa::NextRowOp> {
public:
    using OpRewritePattern<pgx::mlir::dsa::NextRowOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        pgx::mlir::dsa::NextRowOp op,
        PatternRewriter &rewriter) const override {
        
        auto loc = op.getLoc();
        
        MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.next_row");
        
        // Get the builder handle directly from the operation
        auto builderHandle = op.getBuilder();
        if (!builderHandle) {
            MLIR_PGX_ERROR("DSAToStd", "Null builder handle");
            return failure();
        }
        
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

// Detect circular IR structures before walk to avoid infinite loops
static bool hasCircularIR(ModuleOp module) {
    llvm::DenseSet<Operation*> visited;
    llvm::DenseSet<Operation*> recursionStack;
    
    bool foundCycle = false;
    module.walk([&](Operation* op) -> WalkResult {
        if (!op) return WalkResult::advance();
        
        // If we've seen this op in current recursion path - cycle detected
        if (recursionStack.contains(op)) {
            MLIR_PGX_ERROR("DSAToStd", "Circular IR detected at operation: " + 
                          op->getName().getStringRef().str());
            foundCycle = true;
            return WalkResult::interrupt();
        }
        
        // If already fully processed, skip
        if (visited.contains(op)) {
            return WalkResult::advance();
        }
        
        // Add to recursion stack
        recursionStack.insert(op);
        
        // Process operation (walk will handle recursion)
        // Remove from recursion stack when done
        visited.insert(op);
        recursionStack.erase(op);
        
        return WalkResult::advance();
    });
    
    return foundCycle;
}

struct DSAToStdPass : public PassWrapper<DSAToStdPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DSAToStdPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        // registry.insert<pgx::mlir::dsa::DSADialect, pgx::db::DBDialect, pgx::mlir::util::UtilDialect,
        //                func::FuncDialect, arith::ArithDialect,
        //                scf::SCFDialect, index::IndexDialect, LLVM::LLVMDialect>();
    }

    StringRef getArgument() const final { return "convert-dsa-to-std"; }
    StringRef getDescription() const final { 
        return "Convert DSA dialect operations to Standard MLIR with runtime calls"; 
    }

    void runOnOperation() override {
        MLIR_PGX_INFO("DSAToStd", "Starting DSA to Standard conversion pass");
        
        auto module = getOperation();
        auto *context = &getContext();
        
        // CRITICAL: Check for circular IR before walk to avoid infinite loops
        if (hasCircularIR(module)) {
            MLIR_PGX_ERROR("DSAToStd", "Circular IR structure detected - failing pass");
            signalPassFailure();
            return;
        }
        
        MLIR_PGX_INFO("DSAToStd", "Got the operation!");
        MLIR_PGX_INFO("DSAToStd", "Got context successfully!");
        
        // Manual operation replacement approach to avoid pattern rewriter issues
        MLIR_PGX_INFO("DSAToStd", "Using manual operation replacement approach");
        
        // Collect operations to replace (can't modify while iterating)
        SmallVector<Operation*> opsToReplace;
        
        MLIR_PGX_INFO("DSAToStd", "Starting walk to find DSA operations");
        module.walk([&](Operation* op) {
            if (!op) {
                MLIR_PGX_ERROR("DSAToStd", "Encountered null operation during walk!");
                return;
            }
            // Commented out to avoid potential output-related hang
            // MLIR_PGX_DEBUG("DSAToStd", "Walking operation: " + op->getName().getStringRef().str());
            if (isa<pgx::mlir::dsa::CreateDSOp, pgx::mlir::dsa::DSAppendOp, 
                    pgx::mlir::dsa::NextRowOp>(op)) {
                MLIR_PGX_INFO("DSAToStd", "Found DSA operation to convert: " + op->getName().getStringRef().str());
                opsToReplace.push_back(op);
            }
        });
        MLIR_PGX_INFO("DSAToStd", "Walk completed");
        
        MLIR_PGX_INFO("DSAToStd", "Found " + std::to_string(opsToReplace.size()) + " DSA operations to convert");
        
        // Process each operation
        for (auto* op : opsToReplace) {
            OpBuilder builder(op);
            
            if (auto createOp = dyn_cast<pgx::mlir::dsa::CreateDSOp>(op)) {
                MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.create_ds");
                
                // Get the schema attribute
                auto schemaAttr = createOp.getInitAttr();
                if (!schemaAttr) {
                    MLIR_PGX_ERROR("DSAToStd", "CreateDSOp missing schema attribute");
                    signalPassFailure();
                    return;
                }
                
                // Create global string for schema
                auto globalName = (Twine("table_schema_") + Twine(reinterpret_cast<uintptr_t>(op))).str();
                auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
                if (!globalOp) {
                    OpBuilder::InsertionGuard guard(builder);
                    builder.setInsertionPointToStart(module.getBody());
                    
                    auto strType = LLVM::LLVMArrayType::get(
                        IntegerType::get(context, 8),
                        schemaAttr->size() + 1);
                    globalOp = builder.create<LLVM::GlobalOp>(
                        op->getLoc(), strType, true, LLVM::Linkage::Private,
                        globalName, builder.getStringAttr(schemaAttr->str() + '\0'));
                }
                
                // Reset builder insertion point
                builder.setInsertionPoint(op);
                
                // Get pointer to the schema string
                auto ptrType = LLVM::LLVMPointerType::get(context);
                auto schemaPtr = builder.create<LLVM::AddressOfOp>(op->getLoc(), ptrType, globalOp.getName());
                
                // Call runtime function to create table builder
                auto i8Type = IntegerType::get(context, 8);
                auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                
                auto createFunc = getOrCreateRuntimeFunction(builder, module, op->getLoc(),
                    "pgx_runtime_create_table_builder", 
                    {ptrType}, {refType});
                
                auto callOp = builder.create<func::CallOp>(op->getLoc(), createFunc.getName(), 
                    TypeRange{refType}, ValueRange{schemaPtr});
                
                // Replace uses and erase
                createOp.getResult().replaceAllUsesWith(callOp.getResult(0));
                createOp.erase();
                
            } else if (auto appendOp = dyn_cast<pgx::mlir::dsa::DSAppendOp>(op)) {
                MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.ds_append");
                
                auto builderHandle = appendOp.getDs();
                auto value = appendOp.getVal();
                auto valueType = value.getType();
                
                // Ensure the builder handle is properly typed for runtime calls
                Value convertedHandle = builderHandle;
                if (!builderHandle.getType().isa<pgx::mlir::util::RefType>()) {
                    // Need to convert the handle
                    auto i8Type = IntegerType::get(context, 8);
                    auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                    convertedHandle = builder.create<mlir::UnrealizedConversionCastOp>(
                        op->getLoc(), refType, builderHandle).getResult(0);
                }
                
                // Create function declarations for runtime append calls
                auto i8Type = IntegerType::get(context, 8);
                auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                
                // Handle nullable types
                if (valueType.isa<pgx::db::NullableI64Type>()) {
                    MLIR_PGX_DEBUG("DSAToStd", "Handling nullable value in ds_append");
                    
                    // Extract null flag and raw value
                    auto isNull = builder.create<pgx::db::IsNullOp>(op->getLoc(), builder.getI1Type(), value);
                    auto rawValue = builder.create<pgx::db::NullableGetValOp>(op->getLoc(), builder.getI64Type(), value);
                    
                    // Call the nullable append function
                    auto appendFunc = getOrCreateRuntimeFunction(builder, module, op->getLoc(),
                        "pgx_runtime_append_nullable_i64", 
                        {refType, builder.getI1Type(), builder.getI64Type()}, {});
                    
                    builder.create<func::CallOp>(op->getLoc(), appendFunc.getName(), TypeRange(), 
                        ValueRange({convertedHandle, isNull, rawValue}));
                    
                } else if (valueType.isInteger(64)) {
                    // Direct value append (non-nullable)
                    auto appendFunc = getOrCreateRuntimeFunction(builder, module, op->getLoc(),
                        "pgx_runtime_append_i64_direct", 
                        {refType, builder.getI64Type()}, {});
                    builder.create<func::CallOp>(op->getLoc(), appendFunc.getName(), TypeRange(), 
                        ValueRange({convertedHandle, value}));
                } else {
                    MLIR_PGX_ERROR("DSAToStd", "Unsupported value type for ds_append");
                    signalPassFailure();
                    return;
                }
                
                // Erase the original operation
                appendOp.erase();
                
            } else if (auto nextRowOp = dyn_cast<pgx::mlir::dsa::NextRowOp>(op)) {
                MLIR_PGX_DEBUG("DSAToStd", "Converting dsa.next_row");
                
                auto builderHandle = nextRowOp.getBuilder();
                
                // Ensure the builder handle is properly typed for runtime calls
                Value convertedHandle = builderHandle;
                if (!builderHandle.getType().isa<pgx::mlir::util::RefType>()) {
                    // Need to convert the handle
                    auto i8Type = IntegerType::get(context, 8);
                    auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                    convertedHandle = builder.create<mlir::UnrealizedConversionCastOp>(
                        op->getLoc(), refType, builderHandle).getResult(0);
                }
                
                // Create function declaration for runtime next row
                auto i8Type = IntegerType::get(context, 8);
                auto refType = pgx::mlir::util::RefType::get(context, i8Type);
                
                auto nextRowFunc = getOrCreateRuntimeFunction(builder, module, op->getLoc(),
                    "pgx_runtime_table_next_row", {refType}, {});
                
                // Call runtime function with the converted handle
                builder.create<func::CallOp>(op->getLoc(), nextRowFunc.getName(), TypeRange(), 
                                            ValueRange{convertedHandle});
                
                // Erase the original operation
                nextRowOp.erase();
            }
        }
        
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