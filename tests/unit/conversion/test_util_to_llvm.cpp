#include <gtest/gtest.h>
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx::mlir;

class UtilToLLVMTest : public ::testing::Test {
protected:
    MLIRContext context;
    OpBuilder builder;
    Location loc;
    ModuleOp module;
    
    UtilToLLVMTest() : builder(&context), loc(builder.getUnknownLoc()) {
        // Register required dialects
        context.loadDialect<pgx::mlir::util::UtilDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        
        module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
    }
    
    ~UtilToLLVMTest() {
        if (module) {
            module.erase();
        }
    }
    
    bool applyUtilToLLVMConversion() {
        PassManager pm(&context);
        pm.addPass(pgx::mlir::util::createUtilToLLVMPass());
        
        if (failed(pm.run(module))) {
            PGX_ERROR("UtilToLLVM pass failed");
            return false;
        }
        
        if (failed(verify(module))) {
            PGX_ERROR("Module verification failed after UtilToLLVM pass");
            return false;
        }
        
        return true;
    }
};

// Test util.ref type conversion
TEST_F(UtilToLLVMTest, RefTypeConversion) {
    // Create a function that uses util.ref type
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_ref_type", funcType);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create util.ref<i32> type
    auto i32Type = builder.getI32Type();
    auto refType = pgx::mlir::util::RefType::get(&context, i32Type);
    
    // Create an alloc operation
    auto allocOp = builder.create<pgx::mlir::util::AllocOp>(loc, refType, mlir::Value());
    
    // Add return
    builder.create<func::ReturnOp>(loc);
    
    // Apply conversion
    ASSERT_TRUE(applyUtilToLLVMConversion());
    
    // Verify the alloc was converted to LLVM operations
    bool foundLLVMCall = false;
    func.walk([&](LLVM::CallOp callOp) {
        // Should find a call to palloc
        auto callee = callOp.getCallee();
        if (callee && *callee == "palloc") {
            foundLLVMCall = true;
        }
    });
    
    EXPECT_TRUE(foundLLVMCall) << "Expected to find LLVM call to palloc";
}

// Test util.pack operation lowering
TEST_F(UtilToLLVMTest, PackOpLowering) {
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_pack", funcType);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create some values to pack
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto val1 = builder.create<arith::ConstantOp>(loc, i32Type, 
                                                   builder.getI32IntegerAttr(42));
    auto val2 = builder.create<arith::ConstantOp>(loc, i64Type,
                                                   builder.getI64IntegerAttr(100));
    
    // Create tuple type
    auto tupleType = TupleType::get(&context, {i32Type, i64Type});
    
    // Create pack operation
    auto packOp = builder.create<pgx::mlir::util::PackOp>(loc, tupleType, 
                                               ValueRange{val1, val2});
    
    builder.create<func::ReturnOp>(loc);
    
    // Apply conversion
    ASSERT_TRUE(applyUtilToLLVMConversion());
    
    // Verify pack was converted to LLVM struct operations
    bool foundInsertValue = false;
    func.walk([&](LLVM::InsertValueOp insertOp) {
        foundInsertValue = true;
    });
    
    EXPECT_TRUE(foundInsertValue) << "Expected to find LLVM insertvalue operations";
}

// Test util.get_tuple operation lowering
TEST_F(UtilToLLVMTest, GetTupleOpLowering) {
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_get_tuple", funcType);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a tuple
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto val1 = builder.create<arith::ConstantOp>(loc, i32Type, 
                                                   builder.getI32IntegerAttr(42));
    auto val2 = builder.create<arith::ConstantOp>(loc, i64Type,
                                                   builder.getI64IntegerAttr(100));
    
    auto tupleType = TupleType::get(&context, {i32Type, i64Type});
    auto packOp = builder.create<pgx::mlir::util::PackOp>(loc, tupleType, 
                                               ValueRange{val1, val2});
    
    // Extract element at index 0
    auto getTupleOp = builder.create<pgx::mlir::util::GetTupleOp>(loc, i32Type, 
                                                        packOp.getResult(), 0);
    
    builder.create<func::ReturnOp>(loc);
    
    // Apply conversion
    ASSERT_TRUE(applyUtilToLLVMConversion());
    
    // Verify get_tuple was converted to LLVM extractvalue
    bool foundExtractValue = false;
    func.walk([&](LLVM::ExtractValueOp extractOp) {
        foundExtractValue = true;
    });
    
    EXPECT_TRUE(foundExtractValue) << "Expected to find LLVM extractvalue operation";
}

// Test complete pipeline with all operations
TEST_F(UtilToLLVMTest, CompletePipeline) {
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_complete", funcType);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Allocate memory for a tuple
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto tupleType = TupleType::get(&context, {i32Type, i64Type});
    auto refType = pgx::mlir::util::RefType::get(&context, tupleType);
    
    auto allocOp = builder.create<pgx::mlir::util::AllocOp>(loc, refType, mlir::Value());
    
    // Create values and pack them
    auto val1 = builder.create<arith::ConstantOp>(loc, i32Type, 
                                                   builder.getI32IntegerAttr(42));
    auto val2 = builder.create<arith::ConstantOp>(loc, i64Type,
                                                   builder.getI64IntegerAttr(100));
    auto packOp = builder.create<pgx::mlir::util::PackOp>(loc, tupleType, 
                                               ValueRange{val1, val2});
    
    // Extract first element
    auto getTupleOp = builder.create<pgx::mlir::util::GetTupleOp>(loc, i32Type, 
                                                        packOp.getResult(), 0);
    
    builder.create<func::ReturnOp>(loc);
    
    // Apply conversion
    ASSERT_TRUE(applyUtilToLLVMConversion());
    
    // Verify all operations were converted
    int llvmOpCount = 0;
    func.walk([&](Operation *op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "llvm") {
            llvmOpCount++;
        }
    });
    
    // Should have multiple LLVM operations (call, bitcast, insertvalue, extractvalue, etc.)
    EXPECT_GT(llvmOpCount, 3) << "Expected multiple LLVM operations after conversion";
    
    // Verify no Util operations remain
    bool foundUtilOp = false;
    func.walk([&](Operation *op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "util") {
            foundUtilOp = true;
        }
    });
    
    EXPECT_FALSE(foundUtilOp) << "No Util operations should remain after conversion";
}