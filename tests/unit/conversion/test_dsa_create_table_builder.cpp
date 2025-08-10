//===- test_dsa_create_table_builder.cpp - Test for CreateTableBuilder lowering --===//

#include "gtest/gtest.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "execution/logging.h"

// Forward declare the new pass creation function
namespace mlir {
std::unique_ptr<Pass> createDSAToStdPass();
}

using namespace mlir;

class CreateTableBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<pgx::mlir::util::UtilDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
    }
    
    mlir::MLIRContext context;
};

TEST_F(CreateTableBuilderTest, BasicCreateTableBuilder) {
    const char* mlir_input = R"mlir(
module {
    func.func @test_create() -> !dsa.table_builder<tuple<i64>> {
        %builder = dsa.create_ds ("id:int[64]") -> !dsa.table_builder<tuple<i64>>
        return %builder : !dsa.table_builder<tuple<i64>>
    }
}
)mlir";
    
    auto module = parseSourceString<ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module) << "Failed to parse MLIR";
    
    // Run the new DSA to Standard pass
    PassManager pm(&context);
    pm.addPass(mlir::createDSAToStdPass());
    
    // This should succeed now because we handle type conversions properly
    ASSERT_TRUE(succeeded(pm.run(*module))) << "DSAToStd pass failed";
    
    // Check that dsa.create_ds was converted
    bool foundDSAOp = false;
    bool foundGlobalString = false;
    bool foundAddressOf = false;
    bool foundRuntimeCall = false;
    
    module->walk([&](Operation* op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "dsa") {
            foundDSAOp = true;
        }
        
        if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
            // Check if this is our schema string
            if (globalOp.getSymName().starts_with("table_schema_")) {
                foundGlobalString = true;
                // Verify it contains the schema
                auto value = globalOp.getValueAttr();
                if (value) {
                    auto strAttr = value.dyn_cast<StringAttr>();
                    if (strAttr) {
                        EXPECT_EQ(strAttr.getValue().str(), std::string("id:int[64]\0", 11));
                    }
                }
            }
        }
        
        if (isa<LLVM::AddressOfOp>(op)) {
            foundAddressOf = true;
        }
        
        if (auto callOp = dyn_cast<func::CallOp>(op)) {
            if (callOp.getCallee() == "pgx_runtime_create_table_builder") {
                foundRuntimeCall = true;
                // Verify it takes pointer and returns ref<i8>
                EXPECT_EQ(callOp.getNumOperands(), 1);
                EXPECT_TRUE(callOp.getOperand(0).getType().isa<LLVM::LLVMPointerType>());
                EXPECT_EQ(callOp.getNumResults(), 1);
                EXPECT_TRUE(callOp.getResult(0).getType().isa<pgx::mlir::util::RefType>());
            }
        }
    });
    
    EXPECT_FALSE(foundDSAOp) << "DSA operations should be converted";
    EXPECT_TRUE(foundGlobalString) << "Should create global string for schema";
    EXPECT_TRUE(foundAddressOf) << "Should get address of global string";
    EXPECT_TRUE(foundRuntimeCall) << "Should call runtime create function";
}

TEST_F(CreateTableBuilderTest, CreateTableBuilderNoReturn) {
    // Test without returning the value to ensure no type mismatch
    const char* mlir_input = R"mlir(
module {
    func.func @test_create_no_return() {
        %builder = dsa.create_ds ("name:string;age:int[32]") -> !dsa.table_builder<tuple<i32, i32>>
        return
    }
}
)mlir";
    
    auto module = parseSourceString<ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module) << "Failed to parse MLIR";
    
    PassManager pm(&context);
    pm.addPass(mlir::createDSAToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(*module))) << "DSAToStd pass failed";
    
    // Verify the schema was correctly extracted
    bool foundCorrectSchema = false;
    module->walk([&](LLVM::GlobalOp globalOp) {
        if (globalOp.getSymName().starts_with("table_schema_")) {
            auto value = globalOp.getValueAttr();
            if (value) {
                auto strAttr = value.dyn_cast<StringAttr>();
                if (strAttr && strAttr.getValue().str() == std::string("name:string;age:int[32]\0", 24)) {
                    foundCorrectSchema = true;
                }
            }
        }
    });
    
    EXPECT_TRUE(foundCorrectSchema) << "Schema should be preserved correctly";
}