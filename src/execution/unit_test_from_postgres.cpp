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

// 🎯 SYSTEMATIC PASS ISOLATION: Add StandardToLLVMPass components one by one
extern "C" bool test_dummy_pass_from_postgresql(mlir::ModuleOp real_module) {
    PGX_INFO("🔬 SYSTEMATIC ISOLATION: Testing individual StandardToLLVMPass components");
    
    try {
        auto* context = real_module.getContext();
        
        // TEST 1: Just DataLayoutAnalysis access
        PGX_INFO("🔬 TEST 1: Testing DataLayoutAnalysis access...");
        mlir::PassManager pm1(context);
        pm1.addPass(mlir::createCanonicalizerPass()); // Safe pass to trigger analysis
        if (!mlir::succeeded(pm1.run(real_module))) {
            PGX_ERROR("❌ TEST 1 FAILED: DataLayoutAnalysis access");
            return false;
        }
        PGX_INFO("✅ TEST 1 PASSED: DataLayoutAnalysis access works");
        
        // TEST 2: LLVMTypeConverter creation
        PGX_INFO("🔬 TEST 2: Testing LLVMTypeConverter creation...");
        try {
            // Get data layout analysis (simulate what StandardToLLVMPass does)
            mlir::DataLayout dataLayout(real_module);
            
            mlir::LowerToLLVMOptions options(context, dataLayout);
            mlir::LLVMTypeConverter typeConverter(context, options);
            PGX_INFO("✅ TEST 2 PASSED: LLVMTypeConverter creation works");
        } catch (const std::exception& e) {
            PGX_ERROR("❌ TEST 2 FAILED: LLVMTypeConverter creation: " + std::string(e.what()));
            return false;
        }
        
        // TEST 3: Pattern population (each one individually)
        PGX_INFO("🔬 TEST 3: Testing pattern population...");
        try {
            mlir::DataLayout dataLayout(real_module);
            
            mlir::LowerToLLVMOptions options(context, dataLayout);
            mlir::LLVMTypeConverter typeConverter(context, options);
            
            mlir::RewritePatternSet patterns(context);
            
            PGX_INFO("🔍 Testing Affine→Standard patterns...");
            mlir::populateAffineToStdConversionPatterns(patterns);
            PGX_INFO("✅ Affine→Standard patterns OK");
            
            PGX_INFO("🔍 Testing SCF→ControlFlow patterns...");
            mlir::populateSCFToControlFlowConversionPatterns(patterns);
            PGX_INFO("✅ SCF→ControlFlow patterns OK");
            
            PGX_INFO("🔍 Testing Func→LLVM patterns...");
            mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
            PGX_INFO("✅ Func→LLVM patterns OK");
            
            PGX_INFO("🔍 Testing Util→LLVM patterns... (CRITICAL TEST)");
            mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
            PGX_INFO("✅ Util→LLVM patterns OK");
            
            PGX_INFO("🔍 Testing Arith→LLVM patterns...");
            mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
            PGX_INFO("✅ Arith→LLVM patterns OK");
            
            PGX_INFO("🔍 Testing ControlFlow→LLVM patterns...");
            mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
            PGX_INFO("✅ ControlFlow→LLVM patterns OK");
            
            PGX_INFO("✅ TEST 3 PASSED: All pattern populations work");
        } catch (const std::exception& e) {
            PGX_ERROR("❌ TEST 3 FAILED: Pattern population: " + std::string(e.what()));
            return false;
        }
        
        // TEST 4: LLVMConversionTarget creation
        PGX_INFO("🔬 TEST 4: Testing LLVMConversionTarget creation...");
        try {
            mlir::LLVMConversionTarget target(*context);
            target.addLegalOp<mlir::ModuleOp>();
            PGX_INFO("✅ TEST 4 PASSED: LLVMConversionTarget creation works");
        } catch (const std::exception& e) {
            PGX_ERROR("❌ TEST 4 FAILED: LLVMConversionTarget creation: " + std::string(e.what()));
            return false;
        }
        
        // TEST 5: Systematic pattern combination testing
        PGX_INFO("🔬 TEST 5: Testing applyFullConversion with pattern combinations...");
        
        // Define all pattern types with descriptive names
        struct PatternInfo {
            std::string name;
            std::function<void(mlir::RewritePatternSet&, mlir::LLVMTypeConverter*)> populate;
        };
        
        std::vector<PatternInfo> patternTypes = {
            {"Affine→Standard", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter*) { 
                mlir::populateAffineToStdConversionPatterns(p); 
            }},
            {"SCF→ControlFlow", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter*) { 
                mlir::populateSCFToControlFlowConversionPatterns(p); 
            }},
            {"Func→LLVM", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter* tc) { 
                mlir::populateFuncToLLVMConversionPatterns(*tc, p); 
            }},
            {"Util→LLVM", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter* tc) { 
                mlir::util::populateUtilToLLVMConversionPatterns(*tc, p); 
            }},
            {"Arith→LLVM", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter* tc) { 
                mlir::arith::populateArithToLLVMConversionPatterns(*tc, p); 
            }},
            {"ControlFlow→LLVM", [](mlir::RewritePatternSet& p, mlir::LLVMTypeConverter* tc) { 
                mlir::cf::populateControlFlowToLLVMConversionPatterns(*tc, p); 
            }}
        };

        PGX_INFO("==================================================================================================");
        PGX_INFO("==================================================================================================");
        PGX_INFO("==================================================================================================");

        // EXHAUSTIVE TESTING: Track all results instead of stopping on first failure
        std::vector<std::string> successfulCombinations;
        std::vector<std::string> failedCombinations;  
        std::vector<std::string> crashedCombinations;
        
        // Test all combinations from size 1 to size N
        for (size_t combinationSize = 1; combinationSize <= patternTypes.size(); combinationSize++) {
            PGX_INFO("🔍 TESTING COMBINATIONS OF SIZE " + std::to_string(combinationSize) + ":");
            
            // Generate all combinations of the current size
            std::vector<bool> selector(patternTypes.size(), false);
            std::fill(selector.end() - combinationSize, selector.end(), true);
            
            do {
                // Build pattern name list for this combination
                std::vector<std::string> selectedNames;
                std::vector<size_t> selectedIndices;
                for (size_t i = 0; i < patternTypes.size(); i++) {
                    if (selector[i]) {
                        selectedNames.push_back(patternTypes[i].name);
                        selectedIndices.push_back(i);
                    }
                }
                
                std::string combinationName = "[";
                for (size_t i = 0; i < selectedNames.size(); i++) {
                    if (i > 0) combinationName += " + ";
                    combinationName += selectedNames[i];
                }
                combinationName += "]";
                
                PGX_INFO("  🧪 Testing: " + combinationName);
                
                try {
                    // REPLICATE EXACT mlir_runner ENVIRONMENT: Use PassManager approach
                    mlir::PassManager pm(context);
                    
                    // Create a custom pass that only applies the selected patterns
                    struct TestConversionPass : public mlir::PassWrapper<TestConversionPass, mlir::OperationPass<mlir::ModuleOp>> {
                        std::vector<size_t> selectedIndices;
                        std::vector<PatternInfo>* patternTypes;
                        std::string combinationName;
                        
                        TestConversionPass(const std::vector<size_t>& indices, std::vector<PatternInfo>* types, const std::string& name) 
                            : selectedIndices(indices), patternTypes(types), combinationName(name) {}
                        
                        void runOnOperation() override {
                            auto module = getOperation();
                            auto* context = &getContext();
                            
                            mlir::DataLayout dataLayout(module);
                            mlir::LowerToLLVMOptions options(context, dataLayout);
                            mlir::LLVMTypeConverter typeConverter(context, options);
                            
                            mlir::RewritePatternSet patterns(context);
                            
                            // Add only the selected patterns
                            for (size_t idx : selectedIndices) {
                                (*patternTypes)[idx].populate(patterns, &typeConverter);
                            }
                            
                            mlir::LLVMConversionTarget target(*context);
                            target.addLegalOp<mlir::ModuleOp>();
                            
                            // THE CRITICAL TEST: applyFullConversion via PassManager (like mlir_runner)
                            if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
                                signalPassFailure();
                                return;
                            }
                        }
                    };
                    
                    // Add our test pass to the PassManager (replicating mlir_runner environment)
                    pm.addPass(std::make_unique<TestConversionPass>(selectedIndices, &patternTypes, combinationName));
                    
                    // THE ULTIMATE TEST: pm.run() exactly like mlir_runner does!
                    if (mlir::succeeded(pm.run(real_module))) {
                        PGX_INFO("    ✅ SUCCESS: " + combinationName + " works via PassManager!");
                        successfulCombinations.push_back(combinationName);
                    } else {
                        PGX_ERROR("    ❌ FAILED: " + combinationName + " failed via PassManager");
                        failedCombinations.push_back(combinationName);
                        // CONTINUE TESTING instead of returning - we want to see ALL results!
                    }
                    
                } catch (const std::exception& e) {
                    PGX_ERROR("    💥 CRASH: " + combinationName + " crashed: " + std::string(e.what()));
                    crashedCombinations.push_back(combinationName + " (exception: " + std::string(e.what()) + ")");
                    // CONTINUE TESTING - we want to see what else crashes!
                } catch (...) {
                    PGX_ERROR("    💥 CRASH: " + combinationName + " crashed with unknown exception");
                    crashedCombinations.push_back(combinationName + " (unknown exception)");
                    // CONTINUE TESTING - we want to see what else crashes!
                }
                
            } while (std::next_permutation(selector.begin(), selector.end()));
        }
        
        // COMPREHENSIVE RESULTS SUMMARY
        PGX_INFO("📊 EXHAUSTIVE TEST RESULTS SUMMARY:");
        PGX_INFO("✅ SUCCESSFUL COMBINATIONS (" + std::to_string(successfulCombinations.size()) + "):");
        for (const auto& combo : successfulCombinations) {
            PGX_INFO("  " + combo);
        }
        
        PGX_INFO("❌ FAILED COMBINATIONS (" + std::to_string(failedCombinations.size()) + "):");
        for (const auto& combo : failedCombinations) {
            PGX_INFO("  " + combo);
        }
        
        PGX_INFO("💥 CRASHED COMBINATIONS (" + std::to_string(crashedCombinations.size()) + "):");
        for (const auto& combo : crashedCombinations) {
            PGX_INFO("  " + combo);
        }
        
        PGX_INFO("==================================================================================================");
        PGX_INFO("==================================================================================================");
        PGX_INFO("==================================================================================================");
        
        // Return false if any failures/crashes were found
        if (!failedCombinations.empty() || !crashedCombinations.empty()) {
            PGX_ERROR("🎯 FOUND PROBLEMATIC COMBINATIONS! See summary above.");
            return false;
        }
        
        PGX_INFO("🤯 INCREDIBLE: ALL pattern combinations work! This shouldn't happen!");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("🧪 SYSTEMATIC ISOLATION: C++ exception: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("🧪 SYSTEMATIC ISOLATION: Unknown exception");
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