// Unit tests for ReduceOperations.cpp - Comprehensive testing of reduction operation lowering
// This file tests reduce operations that might be adding operations after terminators during aggregation

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/util/UtilDialect.h"
#include "compiler/Dialect/util/UtilOps.h"
#include "compiler/Dialect/util/UtilTypes.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "compiler/Dialect/TupleStream/TupleStreamTypes.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ReduceOperationsTest : public ::testing::Test {
public:
    ReduceOperationsTest() = default;
    
protected:
    
    void SetUp() override {
        context = std::make_unique<MLIRContext>();
        context->loadDialect<subop::SubOperatorDialect>();
        context->loadDialect<util::UtilDialect>();
        context->loadDialect<tuples::TupleStreamDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<func::FuncDialect>();
        
        builder = std::make_unique<OpBuilder>(context.get());
        loc = builder->getUnknownLoc();
        
        // Create module for tests
        module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
    }

    void TearDown() override {
        module.erase();
    }

    // Helper to create a test function context
    func::FuncOp createTestFunction(StringRef name, FunctionType funcType) {
        auto funcOp = builder->create<func::FuncOp>(loc, name, funcType);
        builder->setInsertionPointToStart(funcOp.addEntryBlock());
        return funcOp;
    }

    // Helper to create a reduce operation with a simple region
    subop::ReduceOp createSimpleReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        // For testing, use nullptr for the column pointer - this may cause issues but is simplest for compilation testing
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
        // Create a simple reduction region with addition
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        // Add block arguments for columns and members
        auto i32Type = builder->getI32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(i32Type, loc);
        }
        
        // Create addition operation in the region
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1); // Last argument is the member
        auto addOp = builder->create<arith::AddIOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{addOp.getResult()});
        
        return reduceOp;
    }

    // Helper to create a floating-point reduce operation
    subop::ReduceOp createFloatReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        auto f32Type = builder->getF32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(f32Type, loc);
        }
        
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1);
        auto addFOp = builder->create<arith::AddFOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{addFOp});
        
        return reduceOp;
    }

    // Helper to create a bitwise OR reduce operation
    subop::ReduceOp createBitwiseOrReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        auto i32Type = builder->getI32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(i32Type, loc);
        }
        
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1);
        auto orOp = builder->create<arith::OrIOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{orOp});
        
        return reduceOp;
    }

    std::unique_ptr<MLIRContext> context;
    std::unique_ptr<OpBuilder> builder;
    Location loc = UnknownLoc::get(context.get());
    ModuleOp module;
    
    // Helper to create a simple tuple stream
    Value createSimpleTupleStream() {
        auto tupleStreamType = tuples::TupleStreamType::get(context.get());
        return builder->create<util::UndefOp>(loc, tupleStreamType);
    }
};