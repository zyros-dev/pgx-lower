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
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "execution/logging.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <functional>
#include <algorithm>

// Test the REAL module from the pipeline - not synthetic data
extern "C" bool test_real_module_from_postgresql(mlir::ModuleOp real_module) {
    PGX_INFO("🧪 EXPERIMENT: Testing REAL pipeline module from within PostgreSQL!");
    PGX_INFO("🎯 Using the EXACT same module that crashes in the original pipeline");
    
    try {
        // Get the context from the real module (not creating fresh one)
        auto* context = real_module.getContext();
        
        PGX_INFO("📋 Real module statistics before test:");
        std::map<std::string, int> dialectCounts;
        real_module.walk([&](mlir::Operation* op) {
            if (op->getDialect()) {
                dialectCounts[op->getDialect()->getNamespace().str()]++;
            }
        });
        
        for (const auto& [dialect, count] : dialectCounts) {
            PGX_INFO("  - " + dialect + ": " + std::to_string(count));
        }

        // 🔍 DEEPWIKI DEBUGGING: Dump real module IR before crash
        PGX_INFO("🔍 DEEPWIKI DEBUG: Dumping real module IR to /tmp/real_module.mlir");
        std::string moduleStr;
        llvm::raw_string_ostream stream(moduleStr);
        real_module.print(stream);
        
        std::ofstream file("/tmp/real_module.mlir");
        if (file.is_open()) {
            file << moduleStr;
            file.close();
            PGX_INFO("✅ Real module IR dumped successfully");
        } else {
            PGX_ERROR("❌ Failed to dump real module IR");
        }
        
        // 🔍 DEEPWIKI DEBUG: Detailed context inspection
        PGX_INFO("🔍 DEEPWIKI DEBUG: MLIRContext details:");
        PGX_INFO("  - Context ptr: " + std::to_string(reinterpret_cast<uintptr_t>(context)));
        PGX_INFO("  - Module context ptr: " + std::to_string(reinterpret_cast<uintptr_t>(real_module.getContext())));
        PGX_INFO("  - Context threading disabled: " + std::to_string(context->isMultithreadingEnabled() ? 0 : 1));
        
        auto loadedDialects = context->getLoadedDialects();
        PGX_INFO("  - Loaded dialects count: " + std::to_string(loadedDialects.size()));
        for (auto* dialect : loadedDialects) {
            if (dialect) {
                PGX_INFO("    - " + dialect->getNamespace().str());
            }
        }
        
        PGX_INFO("🔥 CRITICAL: Creating NEW PassManager with SAME module");
        mlir::PassManager pm(context);
        
        // 🔍 DEEPWIKI DEBUG: PassManager state before adding passes
        PGX_INFO("🔍 DEEPWIKI DEBUG: PassManager created successfully");
        PGX_INFO("  - PassManager ptr: " + std::to_string(reinterpret_cast<uintptr_t>(&pm)));
        
        pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
        PGX_INFO("🔍 DEEPWIKI DEBUG: StandardToLLVMPass added successfully");
        
        PGX_INFO("🎯 THE ULTIMATE TEST: pm.run() with REAL module in PostgreSQL...");
        PGX_INFO("🔍 ABOUT TO CALL pm.run() - this is where we expect the crash");
        
        // 🎯 THE CRITICAL MOMENT: Same module, fresh PassManager
        if (mlir::succeeded(pm.run(real_module))) {
            PGX_INFO("🤯 INCREDIBLE: Real module pm.run() SUCCEEDED in PostgreSQL!");
            PGX_INFO("🔍 This suggests the issue is PassManager state, not module content");
            return true;
        } else {
            PGX_ERROR("❌ Real module pm.run() failed - but with fresh PassManager!");
            PGX_ERROR("🔍 This suggests the issue is the module content itself");
            return false;
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("🧪 REAL MODULE TEST: C++ exception: " + std::string(e.what()));
        PGX_ERROR("🔍 Same module content crashes even with fresh PassManager");
        return false;
    } catch (...) {
        PGX_ERROR("🧪 REAL MODULE TEST: Unknown exception with real module");
        PGX_ERROR("🔍 Module content itself may be corrupted or problematic");
        return false;
    }
}

// 🎯 EMPTY PASSMANAGER TEST: Does pm.run() crash with NO passes?
extern "C" bool test_empty_passmanager_from_postgresql(mlir::ModuleOp real_module) {
    PGX_INFO("🧪 EMPTY PASSMANAGER TEST: Testing EMPTY PassManager with real module");
    
    try {
        auto* context = real_module.getContext();
        
        PGX_INFO("🔍 Creating COMPLETELY EMPTY PassManager...");
        mlir::PassManager pm(context);
        // NO PASSES ADDED - completely empty!
        
        PGX_INFO("🎯 CRITICAL: Calling pm.run() with NO passes on real module...");
        PGX_INFO("🔍 If this crashes, the issue is pm.run() itself, not our passes");
        
        if (mlir::succeeded(pm.run(real_module))) {
            PGX_INFO("✅ EMPTY PassManager works! Issue is in our StandardToLLVMPass");
            return true;
        } else {
            PGX_INFO("❌ EMPTY PassManager failed - issue is deeper than our passes");
            return false;
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("🧪 EMPTY PASSMANAGER TEST: Exception: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("🧪 EMPTY PASSMANAGER TEST: Unknown exception");
        return false;
    }
}

// Legacy synthetic test (keeping for comparison)
extern "C" bool test_unit_code_from_postgresql() {
    PGX_INFO("🧪 SYNTHETIC TEST: Creating fresh module from scratch in PostgreSQL");
    
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
        
        // Simple operations for testing
        auto constantValue = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 42, 32);
        PGX_INFO("Created synthetic operations in PostgreSQL context");
        
        auto val1 = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 1, 32);
        auto val2 = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 99, 32);
        
        auto tupleType = mlir::TupleType::get(&context, 
            {builder.getI32Type(), builder.getI32Type()});
        auto packOp = builder.create<mlir::util::PackOp>(
            builder.getUnknownLoc(), tupleType, 
            mlir::ValueRange{val1, val2});
        
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        
        // 🔍 DEEPWIKI DEBUGGING: Dump synthetic module IR for comparison
        PGX_INFO("🔍 DEEPWIKI DEBUG: Dumping synthetic module IR to /tmp/synthetic_module.mlir");
        std::string syntheticModuleStr;
        llvm::raw_string_ostream syntheticStream(syntheticModuleStr);
        module.print(syntheticStream);
        
        std::ofstream syntheticFile("/tmp/synthetic_module.mlir");
        if (syntheticFile.is_open()) {
            syntheticFile << syntheticModuleStr;
            syntheticFile.close();
            PGX_INFO("✅ Synthetic module IR dumped successfully");
        } else {
            PGX_ERROR("❌ Failed to dump synthetic module IR");
        }

        mlir::PassManager pm(&context);
        pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
        
        PGX_INFO("🔥 CALLING pm.run(synthetic_module) FROM POSTGRESQL...");
        
        if (mlir::succeeded(pm.run(module))) {
            PGX_INFO("✅ Synthetic module works in PostgreSQL");
            return true;
        } else {
            PGX_ERROR("❌ Synthetic module failed in PostgreSQL");
            return false;
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("🧪 SYNTHETIC TEST: Exception: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("🧪 SYNTHETIC TEST: Unknown exception");
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