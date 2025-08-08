#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

using namespace mlir;

// Minimal test to check if pass runs without segfault
TEST(RelAlgToDBSimpleTest, PassRunsWithoutCrash) {
    PGX_DEBUG("Testing minimal RelAlgToDB pass execution");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    
    // Create module and empty function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_empty", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Just an empty return
    builder.create<func::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass on empty function
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    PGX_DEBUG("Running pass on empty function...");
    LogicalResult result = pm.run(funcOp);
    PGX_DEBUG("Pass completed");
    
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed on empty function";
    
    PGX_DEBUG("Test completed successfully");
}

// Test with just MaterializeOp, no BaseTable
TEST(RelAlgToDBSimpleTest, MaterializeOpAlone) {
    PGX_DEBUG("Testing MaterializeOp without BaseTable");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {tableType});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_materialize_alone", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a dummy tuple stream value (normally from BaseTable)
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    // For this test, we'll use a block argument as the stream source
    auto streamArg = entryBlock->addArgument(tupleStreamType, UnknownLoc::get(&context));
    
    // Create MaterializeOp
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("dummy"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        streamArg,
        columnsArrayAttr
    );
    
    // Return the materialized table
    builder.create<func::ReturnOp>(UnknownLoc::get(&context), materializeOp.getResult());
    
    // Update function type to accept the stream argument
    auto newFuncType = builder.getFunctionType({tupleStreamType}, {tableType});
    funcOp.setType(newFuncType);
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    PGX_DEBUG("Running pass on MaterializeOp alone...");
    LogicalResult result = pm.run(funcOp);
    PGX_DEBUG("Pass completed");
    
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should handle MaterializeOp without BaseTable";
    
    PGX_DEBUG("Test completed successfully");
}