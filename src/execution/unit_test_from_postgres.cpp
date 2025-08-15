#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "execution/logging.h"

// EXACT COPY of the unit test that works perfectly outside PostgreSQL
extern "C" bool test_unit_code_from_postgresql() {
    PGX_INFO("🧪 EXPERIMENT: Running EXACT unit test code from within PostgreSQL!");
    
    try {
        mlir::MLIRContext context;
        mlir::OpBuilder builder(&context);
        
        // Register required dialects (same as unit test)
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::memref::MemRefDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        
        // Create module (same as unit test)
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Create a realistic function that mirrors what Phase 3b produces  
        builder.setInsertionPointToEnd(module.getBody());
        
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "query_main", funcType);
        
        auto* block = func.addEntryBlock();
        builder.setInsertionPointToEnd(block);
        
        // THE EXACT SAME OPERATIONS FROM THE UNIT TEST:
        
        // 1. Simple arithmetic - forms the basis of query operations
        auto constantValue = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 42, 32);
        PGX_INFO("Created arithmetic operations in PostgreSQL context");
        
        // 2. Create a util.pack operation with real values (similar to tuple operations)
        auto val1 = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 1, 32);
        auto val2 = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 99, 32);
        
        auto tupleType = mlir::TupleType::get(&context, 
            {builder.getI32Type(), builder.getI32Type()});
        auto packOp = builder.create<mlir::util::PackOp>(
            builder.getUnknownLoc(), tupleType, 
            mlir::ValueRange{val1, val2});
        PGX_INFO("Created util.pack in PostgreSQL context");
        
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        
        PGX_INFO("💀 SAME MODULE THAT WORKS IN UNIT TEST, NOW IN POSTGRESQL!");
        PGX_INFO("🎯 About to call pm.run() - THE MOMENT OF TRUTH!");
        
        // THE EXACT SAME PASS MANAGER SETUP FROM UNIT TEST:
        mlir::PassManager pm(&context);
        pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
        
        PGX_INFO("🔥 CALLING pm.run(module) FROM POSTGRESQL CONTEXT...");
        
        // 🎯 THE CRITICAL MOMENT: Same code, different environment
        if (mlir::succeeded(pm.run(module))) {
            PGX_INFO("🤯 SHOCKING: pm.run() succeeded in PostgreSQL context!");
            PGX_INFO("This would DISPROVE the environment incompatibility theory");
            return true;
        } else {
            PGX_ERROR("❌ pm.run() failed in PostgreSQL context");
            PGX_ERROR("But unit test shows the MLIR code is correct");
            return false;
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("🧪 EXPERIMENT RESULT: C++ exception in PostgreSQL: " + std::string(e.what()));
        PGX_ERROR("Same code works in unit test but crashes in PostgreSQL");
        PGX_ERROR("PROOF: Environment incompatibility confirmed");
        return false;
    } catch (...) {
        PGX_ERROR("🧪 EXPERIMENT RESULT: Unknown exception in PostgreSQL");
        PGX_ERROR("Same code works in unit test but crashes in PostgreSQL");  
        PGX_ERROR("PROOF: Environment incompatibility confirmed");
        return false;
    }
}

// Alternative: Test just the PassManager creation without running it
extern "C" bool test_passmanager_creation_only() {
    PGX_INFO("🔬 ISOLATING: Testing just PassManager creation in PostgreSQL");
    
    try {
        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        
        PGX_INFO("Creating PassManager...");
        mlir::PassManager pm(&context);
        
        PGX_INFO("Adding pass...");
        pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
        
        PGX_INFO("✅ PassManager creation succeeded in PostgreSQL");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("❌ PassManager creation failed: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("❌ PassManager creation failed with unknown exception");
        return false;
    }
}