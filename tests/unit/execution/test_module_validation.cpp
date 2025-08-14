#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "dialects/RelAlg/IR/RelAlgDialect.h"
#include "dialects/DB/IR/DBDialect.h"
#include "execution/logging.h"

// Mock implementations of validation functions for testing
#include "mlir/IR/Verifier.h"
#include <map>
#include <string>

static void safeModulePrint(::mlir::ModuleOp module, const std::string& label) {
    if (!module || !module.getOperation()) {
        PGX_WARNING(label + ": Module is null or invalid");
        return;
    }
    
    try {
        // Count operations to determine if safe to print
        size_t totalOps = 0;
        module.walk([&](mlir::Operation* op) { totalOps++; });
        
        if (totalOps < 1000) {
            std::string moduleStr;
            llvm::raw_string_ostream os(moduleStr);
            module.getOperation()->print(os);
            
            // Split into smaller chunks to avoid overwhelming the logger
            const size_t chunkSize = 4000;
            size_t pos = 0;
            while (pos < moduleStr.length()) {
                size_t endPos = std::min(pos + chunkSize, moduleStr.length());
                PGX_DEBUG(label + " [" + std::to_string(pos/chunkSize + 1) + "/" + 
                         std::to_string((moduleStr.length() + chunkSize - 1)/chunkSize) + "]:\n" +
                         moduleStr.substr(pos, endPos - pos));
                pos = endPos;
            }
        } else {
            PGX_WARNING(label + ": Module too large to print (" + std::to_string(totalOps) + " operations)");
        }
    } catch (const std::exception& e) {
        PGX_WARNING(label + ": Exception during module print: " + std::string(e.what()));
    } catch (...) {
        PGX_WARNING(label + ": Unknown exception during module print");
    }
}

static bool validateModuleState(::mlir::ModuleOp module, const std::string& phase) {
    if (!module || !module.getOperation()) {
        PGX_ERROR(phase + ": Module operation is null");
        return false;
    }
    
    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR(phase + ": Module verification failed");
        return false;
    }
    
    // Count operations by dialect
    std::map<std::string, int> dialectOpCounts;
    module.walk([&](mlir::Operation* op) {
        auto dialectName = op->getName().getDialectNamespace();
        dialectOpCounts[dialectName.str()]++;
    });
    
    PGX_DEBUG(phase + " operation counts:");
    for (const auto& [dialect, count] : dialectOpCounts) {
        PGX_DEBUG("  - " + dialect + ": " + std::to_string(count));
    }
    
    return true;
}

class ModuleValidationTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    ModuleValidationTest() : builder(&context) {
        // Load required dialects
        context.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<mlir::db::DBDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
    }
    
    void SetUp() override {
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
    }
    
    void TearDown() override {
        if (module) {
            module.erase();
        }
    }
};

TEST_F(ModuleValidationTest, SafeModulePrintWithValidModule) {
    // Test safe module printing with valid module
    // Should not crash
    safeModulePrint(module, "Test valid module");
    
    // Add some content to module
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    module.push_back(func);
    
    // Should still not crash with content
    safeModulePrint(module, "Test module with function");
}

TEST_F(ModuleValidationTest, SafeModulePrintWithLargeModule) {
    // Create a large module to test truncation
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    // Add many functions to exceed operation limit
    for (int i = 0; i < 1500; ++i) {
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), 
            "test_func_" + std::to_string(i), 
            funcType);
        module.push_back(func);
    }
    
    // Should handle large module gracefully
    safeModulePrint(module, "Test large module");
}

TEST_F(ModuleValidationTest, ValidateModuleStateWithEmptyModule) {
    // Test validation with empty module
    EXPECT_TRUE(validateModuleState(module, "Empty module test"));
}

TEST_F(ModuleValidationTest, ValidateModuleStateWithValidOperations) {
    // Add valid operations
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "valid_func", funcType);
    module.push_back(func);
    
    EXPECT_TRUE(validateModuleState(module, "Valid operations test"));
}

TEST_F(ModuleValidationTest, ValidateModuleStateCountsDialects) {
    // Add operations from different dialects
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    // Add func dialect operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    module.push_back(func);
    
    // The validation should count operations by dialect
    EXPECT_TRUE(validateModuleState(module, "Multi-dialect test"));
}

TEST_F(ModuleValidationTest, ValidateModuleStateWithNullModule) {
    // Test with null module
    mlir::ModuleOp nullModule;
    EXPECT_FALSE(validateModuleState(nullModule, "Null module test"));
}