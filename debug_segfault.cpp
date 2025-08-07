#include <gtest/gtest.h>
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDSA/RelAlgToDSA.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

int main() {
    MLIRContext context;
    
    PGX_DEBUG("Loading func dialect...");
    context.getOrLoadDialect<func::FuncDialect>();
    
    PGX_DEBUG("Loading arith dialect...");
    context.getOrLoadDialect<arith::ArithDialect>();
    
    PGX_DEBUG("Loading RelAlg dialect...");
    context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
    
    PGX_DEBUG("Loading DSA dialect...");
    context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
    
    PGX_DEBUG("Creating builder...");
    OpBuilder builder(&context);
    
    PGX_DEBUG("Creating module...");
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    PGX_DEBUG("Creating function...");
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_function", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    PGX_DEBUG("Creating BaseTableOp...");
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    PGX_DEBUG("Creating BaseTableOp region...");
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    
    PGX_DEBUG("Creating MaterializeOp...");
    SmallVector<Attribute> allColumnAttrs;
    allColumnAttrs.push_back(builder.getStringAttr("id"));
    allColumnAttrs.push_back(builder.getStringAttr("name"));
    auto columnsArrayAttr = builder.getArrayAttr(allColumnAttrs);
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    PGX_DEBUG("Module creation completed successfully!");
    
    // Print the module to see if that causes the crash
    PGX_DEBUG("Trying to print module...");
    module->print(llvm::outs());
    PGX_DEBUG("Module printed successfully");
    
    return 0;
}