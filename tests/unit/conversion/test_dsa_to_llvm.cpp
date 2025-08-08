#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/DSAToLLVM/DSAToLLVM.h"

#include <sstream>

namespace {

class DSAToLLVMConversionTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    DSAToLLVMConversionTest() : builder(&context) {
        // Register necessary dialects
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        
        // Create a module
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(module.getBody());
    }

    ~DSAToLLVMConversionTest() {
        module.erase();
    }
};

TEST_F(DSAToLLVMConversionTest, ConvertCreateDSOp) {
    // Create a function containing DSA operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_create_ds", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create DSA CreateDS operation
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType);
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify conversion: CreateDSOp should be replaced with LLVM call
    bool foundLLVMCall = false;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        if (callOp.getCallee() && 
            callOp.getCallee()->str() == "pgx_dsa_create_table_builder") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "CreateDSOp should be converted to LLVM call";
    
    // Verify no DSA operations remain
    int dsaOpCount = 0;
    module.walk([&](mlir::Operation *op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "dsa") {
            dsaOpCount++;
        }
    });
    
    EXPECT_EQ(dsaOpCount, 0) << "No DSA operations should remain after conversion";
}

TEST_F(DSAToLLVMConversionTest, ConvertScanSourceOp) {
    // Create a function containing ScanSource operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_scan_source", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create DSA ScanSource operation
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto iterableType = pgx::mlir::dsa::GenericIterableType::get(
        &context, tupleType, "postgresql_scan");
    
    auto tableDesc = builder.getStringAttr("{\"type\":\"postgresql_table\"}");
    auto scanOp = builder.create<pgx::mlir::dsa::ScanSourceOp>(
        builder.getUnknownLoc(), iterableType, tableDesc);
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify conversion: ScanSourceOp should be replaced with LLVM call
    bool foundLLVMCall = false;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        if (callOp.getCallee() && 
            callOp.getCallee()->str() == "pgx_dsa_scan_source") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "ScanSourceOp should be converted to LLVM call";
}

TEST_F(DSAToLLVMConversionTest, DISABLED_ConvertFinalizeOp) {
    // Create a function containing Finalize operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_finalize", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create table builder first
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType);
    
    // Create DSA Finalize operation
    auto tableType = pgx::mlir::dsa::TableType::get(&context, tupleType);
    auto finalizeOp = builder.create<pgx::mlir::dsa::FinalizeOp>(
        builder.getUnknownLoc(), tableType, createDSOp.getResult());
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify conversion: FinalizeOp should be replaced with LLVM call
    bool foundLLVMCall = false;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        if (callOp.getCallee() && 
            callOp.getCallee()->str() == "pgx_dsa_finalize_table") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "FinalizeOp should be converted to LLVM call";
}

TEST_F(DSAToLLVMConversionTest, DISABLED_ConvertCompleteFlow) {
    // Test complete DSA flow: CreateDS -> DSAppend -> NextRow -> Finalize
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_complete_flow", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create table builder
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType);
    
    // Append a value
    auto value = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(), createDSOp.getResult(), 
        mlir::ValueRange{value.getResult()});
    
    // Next row
    builder.create<pgx::mlir::dsa::NextRowOp>(
        builder.getUnknownLoc(), createDSOp.getResult());
    
    // Finalize
    auto tableType = pgx::mlir::dsa::TableType::get(&context, tupleType);
    auto finalizeOp = builder.create<pgx::mlir::dsa::FinalizeOp>(
        builder.getUnknownLoc(), tableType, createDSOp.getResult());
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify all operations were converted
    std::set<std::string> expectedCalls = {
        "pgx_dsa_create_table_builder",
        "pgx_dsa_append_i32",
        "pgx_dsa_next_row",
        "pgx_dsa_finalize_table"
    };
    
    std::set<std::string> foundCalls;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        if (callOp.getCallee()) {
            foundCalls.insert(callOp.getCallee()->str());
        }
    });
    
    for (const auto& expectedCall : expectedCalls) {
        EXPECT_TRUE(foundCalls.count(expectedCall) > 0) 
            << "Expected LLVM call to " << expectedCall;
    }
}

TEST_F(DSAToLLVMConversionTest, DISABLED_TypeConversion) {
    // Test that DSA types are properly converted to LLVM types
    mlir::pgx_conversion::DSAToLLVMTypeConverter converter(&context);
    
    // Test TableType conversion
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableType = pgx::mlir::dsa::TableType::get(&context, tupleType);
    auto convertedTableType = converter.convertType(tableType);
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(convertedTableType))
        << "TableType should convert to LLVM pointer";
    
    // Test TableBuilderType conversion
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    auto convertedBuilderType = converter.convertType(tableBuilderType);
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(convertedBuilderType))
        << "TableBuilderType should convert to LLVM pointer";
    
    // Test GenericIterableType conversion
    auto iterableType = pgx::mlir::dsa::GenericIterableType::get(
        &context, tupleType, "test_iterator");
    auto convertedIterableType = converter.convertType(iterableType);
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(convertedIterableType))
        << "GenericIterableType should convert to LLVM pointer";
}

} // namespace