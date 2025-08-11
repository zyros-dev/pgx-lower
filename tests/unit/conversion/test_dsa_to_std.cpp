//===- test_dsa_to_std.cpp - Unit tests for DSA to Standard lowering -------===//

#include "gtest/gtest.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSADialect.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBTypes.h"
#include "pgx_lower/mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "execution/logging.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace llvm;

class DSAToStdTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("Setting up DSAToStd test");
        
        // Register all required dialects
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<memref::MemRefDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }
    
    void runDSAToStdPass(mlir::ModuleOp module) {
        mlir::PassManager pm(&context);
        pm.addPass(mlir::createDSAToStdPass());
        
        ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    }
    
    mlir::MLIRContext context;
};

// Test conversion of dsa.create_ds for TableBuilder
TEST_F(DSAToStdTest, CreateDSTableBuilder) {
    const char* mlir_input = R"mlir(
        module {
            func.func @test_create_table_builder() {
                dsa.create_ds ("id:int[64]") -> !dsa.table_builder<tuple<i64>>
                return
            }
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module);
    
    PGX_DEBUG("Running DSAToStd pass on create_ds test");
    runDSAToStdPass(*module);
    
    // Verify the operation was converted
    bool foundGlobalString = false;
    bool foundRuntimeCall = false;
    bool foundAddressOf = false;
    
    module->walk([&](mlir::Operation* op) {
        if (auto globalOp = dyn_cast<mlir::LLVM::GlobalOp>(op)) {
            PGX_DEBUG("Found LLVM global string for schema");
            foundGlobalString = true;
        }
        
        if (auto addressOp = dyn_cast<mlir::LLVM::AddressOfOp>(op)) {
            PGX_DEBUG("Found LLVM addressof operation");
            foundAddressOf = true;
        }
        
        if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
            if (callOp.getCallee() == "pgx_runtime_create_table_builder") {
                PGX_DEBUG("Found runtime create_table_builder call");
                foundRuntimeCall = true;
                
                // Verify the result type is util.ref<i8>
                EXPECT_EQ(callOp.getNumResults(), 1);
                auto resultType = callOp.getResult(0).getType();
                EXPECT_TRUE(resultType.isa<mlir::util::RefType>());
            }
        }
    });
    
    EXPECT_TRUE(foundGlobalString) << "Expected LLVM global for schema string";
    EXPECT_TRUE(foundAddressOf) << "Expected LLVM addressof to get schema pointer";
    EXPECT_TRUE(foundRuntimeCall) << "Expected runtime call to create_table_builder";
}

// Test conversion of dsa.ds_append with tuple (converted nullable value)
TEST_F(DSAToStdTest, DSAppendNullable) {
    const char* mlir_input = R"mlir(
        module {
            func.func @test_ds_append_nullable(%builder: !dsa.table_builder<tuple<i64>>, 
                                               %val: tuple<i64, i1>) {
                dsa.ds_append %builder : !dsa.table_builder<tuple<i64>>, %val : tuple<i64, i1>
                return
            }
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module);
    
    runDSAToStdPass(*module);
    
    // Verify tuple handling
    bool foundGetTuple = false;
    bool foundAppendCall = false;
    
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "util.get_tuple") {
            PGX_DEBUG("Found util.get_tuple operation");
            foundGetTuple = true;
        }
        
        if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
            if (callOp.getCallee() == "pgx_runtime_append_nullable_i64") {
                PGX_DEBUG("Found runtime append_nullable_i64 call");
                foundAppendCall = true;
                
                // Verify arguments: builder handle, is_null flag, raw value
                EXPECT_EQ(callOp.getNumOperands(), 3);
            }
        }
    });
    
    EXPECT_TRUE(foundGetTuple) << "Expected util.get_tuple for tuple extraction";
    EXPECT_TRUE(foundAppendCall) << "Expected runtime call to append_nullable_i64";
}

// Test conversion of dsa.next_row
TEST_F(DSAToStdTest, NextRow) {
    const char* mlir_input = R"mlir(
        module {
            func.func @test_next_row(%builder: !dsa.table_builder<tuple<i64>>) {
                dsa.next_row %builder : !dsa.table_builder<tuple<i64>>
                return
            }
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module);
    
    runDSAToStdPass(*module);
    
    // Verify next_row conversion
    bool foundNextRowCall = false;
    
    module->walk([&](mlir::Operation* op) {
        if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
            if (callOp.getCallee() == "pgx_runtime_table_next_row") {
                PGX_DEBUG("Found runtime table_next_row call");
                foundNextRowCall = true;
                
                // Verify single argument: builder handle
                EXPECT_EQ(callOp.getNumOperands(), 1);
            }
        }
    });
    
    EXPECT_TRUE(foundNextRowCall) << "Expected runtime call to table_next_row";
}

// Test complete pipeline with dsa operations
TEST_F(DSAToStdTest, CompletePipeline) {
    const char* mlir_input = R"mlir(
module {
  func.func @test_complete_pipeline() {
    %builder = dsa.create_ds ("id:int[64]") -> !dsa.table_builder<tuple<i64>>
    %c42_i64 = arith.constant 42 : i64
    %false = arith.constant false : i1
    %nullable = util.pack %c42_i64, %false : (i64, i1) -> tuple<i64, i1>
    dsa.ds_append %builder : !dsa.table_builder<tuple<i64>>, %nullable : tuple<i64, i1>
    dsa.next_row %builder : !dsa.table_builder<tuple<i64>>
    return
  }
}
)mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_input, &context);
    ASSERT_TRUE(module);
    
    runDSAToStdPass(*module);
    
    // Count operations
    int dsaOpCount = 0;
    int runtimeCallCount = 0;
    
    module->walk([&](mlir::Operation* op) {
        // Check for any remaining DSA operations
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
            dsaOpCount++;
        }
        
        // Count runtime calls
        if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
            auto callee = callOp.getCallee();
            if (callee.starts_with("pgx_runtime_")) {
                runtimeCallCount++;
            }
        }
    });
    
    EXPECT_EQ(dsaOpCount, 0) << "All DSA operations should be converted";
    EXPECT_GE(runtimeCallCount, 3) << "Should have runtime calls for create, append, next_row";
    
    PGX_INFO("DSAToStd complete pipeline test passed successfully");
}