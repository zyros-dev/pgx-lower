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

// Test CreateDSOp conversion (restored in Phase 4d-1)
TEST_F(DSAToLLVMConversionTest, ConvertCreateDSOp) {
    // Create a function containing CreateDS operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_create_ds", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create DSA CreateDS operation for TableBuilder
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto schemaAttr = builder.getStringAttr("id:int[32]");
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, schemaAttr);
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify conversion: CreateDSOp should be replaced with LLVM call
    bool foundLLVMCall = false;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        auto callee = callOp.getCalleeAttr();
        if (callee && callee.getValue() == "pgx_runtime_create_table_builder") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "CreateDSOp should be converted to LLVM call";
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
        auto callee = callOp.getCalleeAttr();
        if (callee && callee.getValue() == "pgx_dsa_scan_source") {
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
    
    // Create a table builder first
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto schemaAttr = builder.getStringAttr("id:int[32]");
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, schemaAttr);
    
    // Create Finalize operation
    auto tableType = pgx::mlir::dsa::TableType::get(&context);
    auto finalizeOp = builder.create<pgx::mlir::dsa::FinalizeOp>(
        builder.getUnknownLoc(), tableType, createDSOp.getDs());
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify conversion: FinalizeOp should be replaced with LLVM call
    bool foundLLVMCall = false;
    module.walk([&](mlir::LLVM::CallOp callOp) {
        auto callee = callOp.getCalleeAttr();
        if (callee && callee.getValue() == "pgx_runtime_table_finalize") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "FinalizeOp should be converted to LLVM call";
}

TEST_F(DSAToLLVMConversionTest, DISABLED_ConvertCompleteFlow) {
    // Create a function with complete table building flow
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_complete_flow", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create table builder
    auto i32Type = builder.getI32Type();
    auto f64Type = builder.getF64Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type, f64Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto schemaAttr = builder.getStringAttr("id:int[32];value:float[64]");
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDSOp>(
        builder.getUnknownLoc(), tableBuilderType, schemaAttr);
    
    // Append values
    auto i32Val = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    auto f64Val = builder.create<mlir::arith::ConstantFloatOp>(
        builder.getUnknownLoc(), llvm::APFloat(3.14), f64Type);
    auto trueVal = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 1, 1);
    
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(), createDSOp.getDs(), i32Val.getResult(), trueVal.getResult());
    builder.create<pgx::mlir::dsa::DSAppendOp>(
        builder.getUnknownLoc(), createDSOp.getDs(), f64Val.getResult(), trueVal.getResult());
    
    // Next row
    builder.create<pgx::mlir::dsa::NextRowOp>(
        builder.getUnknownLoc(), createDSOp.getDs());
    
    // Finalize
    auto tableType = pgx::mlir::dsa::TableType::get(&context);
    builder.create<pgx::mlir::dsa::FinalizeOp>(
        builder.getUnknownLoc(), tableType, createDSOp.getDs());
    
    // Return from function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Apply DSA to LLVM conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createDSAToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify all operations were converted
    int dsaOpsCount = 0;
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
            dsaOpsCount++;
        }
    });
    
    EXPECT_EQ(dsaOpsCount, 0) << "All DSA operations should be converted";
}

TEST_F(DSAToLLVMConversionTest, DISABLED_TypeConversion) {
    // Test DSA type conversion
    mlir::pgx_conversion::DSAToLLVMTypeConverter typeConverter(&context);
    
    // Test TableBuilder type conversion
    auto i32Type = builder.getI32Type();
    auto tupleType = mlir::TupleType::get(&context, {i32Type});
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto convertedTableBuilder = typeConverter.convertType(tableBuilderType);
    ASSERT_TRUE(convertedTableBuilder);
    EXPECT_TRUE(convertedTableBuilder.isa<mlir::LLVM::LLVMPointerType>())
        << "TableBuilder should convert to LLVM pointer";
    
    // Test Table type conversion
    auto tableType = pgx::mlir::dsa::TableType::get(&context);
    auto convertedTable = typeConverter.convertType(tableType);
    ASSERT_TRUE(convertedTable);
    EXPECT_TRUE(convertedTable.isa<mlir::LLVM::LLVMPointerType>())
        << "Table should convert to LLVM pointer";
}

} // namespace