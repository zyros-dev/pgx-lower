// Simplified unit tests for Lookup operations
// Tests basic lookup operations compilation and functionality

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

// Simple test for basic lookup operation compilation
TEST(LookupOperationsTest, BasicLookupOpCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_lookup", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create simple state type
    auto stateType = subop::SimpleStateType::get(&context, builder.getI32Type());
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, stateType), Value());
    
    // Create a key for lookup
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI32Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType, 
        stateValue, ValueRange{keyValue});
    
    // Add terminator to the function
    builder.create<func::ReturnOp>(loc);
    
    // Verify the lookup operation was created
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(lookupOp.getKeys().size() == 1);
    
    // Verify proper termination
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_TRUE(terminator->hasTrait<OpTrait::IsTerminator>());
    
    PGX_INFO("Basic lookup operation test completed successfully");
    
    module.erase();
}

// Test for lookup operation with hash map state
TEST(LookupOperationsTest, HashMapLookupCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_hashmap_lookup", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create hash map type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMapType = subop::HashmapType::get(&context, keyMembers, valueMembers, false);
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, hashMapType), Value());
    
    // Create a key for lookup
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 123, 32);
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType, 
        stateValue, ValueRange{keyValue});
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify the lookup operation was created
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(lookupOp.getKeys().size() == 1);
    EXPECT_TRUE(isa<util::RefType>(stateValue.getType()));
    
    PGX_INFO("Hash map lookup test completed successfully");
    
    module.erase();
}

// Test for insert operation
TEST(LookupOperationsTest, BasicInsertOpCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_insert", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create hash map type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMapType = subop::HashmapType::get(&context, keyMembers, valueMembers, false);
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, hashMapType), Value());
    
    // Create key and value for insert
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 456, 32);
    auto valueValue = builder.create<arith::ConstantIntOp>(loc, 789, 64);
    
    // Create column mapping for insert
    auto columnMapping = ArrayAttr::get(&context, {
        IntegerAttr::get(builder.getI32Type(), 0),
        IntegerAttr::get(builder.getI32Type(), 1)
    });
    
    // Create insert operation
    auto insertOp = builder.create<subop::InsertOp>(loc, stateValue,
        columnMapping, ValueRange{keyValue, valueValue});
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify the insert operation was created
    EXPECT_TRUE(insertOp);
    EXPECT_TRUE(insertOp.getColumns().size() == 2);
    
    PGX_INFO("Basic insert operation test completed successfully");
    
    module.erase();
}

// Test for lookup-or-insert operation
TEST(LookupOperationsTest, LookupOrInsertOpCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<tuplestream::TupleStreamDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_lookup_or_insert", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create hash map type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMapType = subop::HashmapType::get(&context, keyMembers, valueMembers, false);
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, hashMapType), Value());
    
    // Create a key for lookup-or-insert
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 333, 32);
    
    // Create lookup-or-insert operation
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOrInsertOp = builder.create<subop::LookupOrInsertOp>(loc, refType,
        stateValue, ValueRange{keyValue});
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify the lookup-or-insert operation was created
    EXPECT_TRUE(lookupOrInsertOp);
    EXPECT_TRUE(lookupOrInsertOp.getKeys().size() == 1);
    
    PGX_INFO("Lookup-or-insert operation test completed successfully");
    
    module.erase();
}

// Test for external hash index lookup
TEST(LookupOperationsTest, ExternalHashIndexLookupCreation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_external_hash_lookup", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create external hash index type
    auto extIndexType = subop::ExternalHashIndexType::get(&context);
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, extIndexType), Value());
    
    // Create keys for lookup
    auto key1 = builder.create<arith::ConstantIntOp>(loc, 111, 32);
    auto key2 = builder.create<arith::ConstantIntOp>(loc, 222, 32);
    
    // Create lookup operation with multiple keys
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{key1, key2});
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify the lookup operation was created
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getKeys().size(), 2);
    EXPECT_TRUE(isa<subop::ExternalHashIndexType>(extIndexType));
    
    PGX_INFO("External hash index lookup test completed successfully");
    
    module.erase();
}

// Test for control flow with lookup operations
TEST(LookupOperationsTest, LookupWithControlFlow) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_lookup_control_flow", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create simple state type
    auto stateType = subop::SimpleStateType::get(&context, builder.getI32Type());
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, stateType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 100, 32);
    auto refType = util::RefType::get(&context, builder.getI32Type());
    
    // Create if-then structure around lookup
    auto condValue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder.create<scf::IfOp>(loc, condValue,
        [&](OpBuilder& b, Location loc) {
            auto lookupOp = b.create<subop::LookupOp>(loc, refType,
                stateValue, ValueRange{keyValue});
            b.create<scf::YieldOp>(loc);
        });
    
    // Add function terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify control flow structure
    EXPECT_TRUE(ifOp);
    EXPECT_EQ(ifOp.getThenRegion().getBlocks().size(), 1);
    
    // Verify the block is properly terminated
    auto& thenBlock = ifOp.getThenRegion().front();
    EXPECT_TRUE(thenBlock.mightHaveTerminator());
    
    PGX_INFO("Lookup with control flow test completed successfully");
    
    module.erase();
}