#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "execution/logging.h"

class RelAlgToDBTest : public ::testing::Test {
protected:
    std::unique_ptr<mlir::MLIRContext> context;
    
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        context->loadDialect<mlir::relalg::RelAlgDialect>();
        context->loadDialect<mlir::db::DBDialect>();
        context->loadDialect<mlir::dsa::DSADialect>();
        context->loadDialect<mlir::util::UtilDialect>();
    }
};

TEST_F(RelAlgToDBTest, ConvertMaterializeWithBaseTable) {
    PGX_DEBUG("TEST: Creating module with MaterializeOp and BaseTableOp");
    
    mlir::OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    
    // Create module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(context.get());
    auto tableMetaData = mlir::relalg::TableMetaDataAttr::get(
        context.get(),
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(12345)
    );
    auto columnsAttr = builder.getDictionaryAttr({});
    
    auto baseTableOp = builder.create<mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder.getStringAttr("test|oid:12345"),
        tableMetaData,
        columnsAttr
    );
    
    // Create MaterializeOp
    auto tableType = mlir::dsa::TableType::get(context.get());
    auto columnRefs = builder.getArrayAttr({});
    auto columnNames = builder.getArrayAttr({});
    
    auto materializeOp = builder.create<mlir::relalg::MaterializeOp>(
        loc,
        tableType,
        baseTableOp.getResult(),
        columnRefs,
        columnNames
    );
    
    // Create return (for now, just return a constant)
    auto constantOp = builder.create<mlir::arith::ConstantIntOp>(loc, 42, 32);
    builder.create<mlir::func::ReturnOp>(loc, constantOp.getResult());
    
    // Print module before conversion
    PGX_DEBUG("TEST: Module before RelAlg→DB conversion:");
    std::string moduleStr;
    llvm::raw_string_ostream os(moduleStr);
    module.print(os);
    PGX_DEBUG(moduleStr);
    
    // Run RelAlg→DB pass
    mlir::PassManager pm(context.get());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createLowerToDBPass());
    
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result)) << "RelAlg→DB pass failed";
    
    // Print module after conversion
    moduleStr.clear();
    module.print(os);
    PGX_DEBUG("TEST: Module after RelAlg→DB conversion:");
    PGX_DEBUG(moduleStr);
    
    // Verify MaterializeOp was replaced with DSA operations
    bool foundDSAOps = false;
    bool foundRelAlgOps = false;
    
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
            foundDSAOps = true;
            PGX_DEBUG("TEST: Found DSA operation: " + op->getName().getStringRef().str());
        }
        if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            foundRelAlgOps = true;
            PGX_DEBUG("TEST: Found remaining RelAlg operation: " + op->getName().getStringRef().str());
        }
    });
    
    EXPECT_TRUE(foundDSAOps) << "Expected DSA operations after conversion";
    // Note: BaseTableOp may remain as it's referenced but not executed
    // The important thing is that MaterializeOp should be gone
    
    bool materializeOpRemains = false;
    module.walk([&](mlir::relalg::MaterializeOp op) {
        materializeOpRemains = true;
    });
    
    EXPECT_FALSE(materializeOpRemains) << "MaterializeOp should be replaced after conversion";
}

TEST_F(RelAlgToDBTest, ConvertEmptyFunction) {
    PGX_DEBUG("TEST: Testing empty function (no RelAlg ops)");
    
    mlir::OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    
    // Create module with empty function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Just return a constant
    auto constantOp = builder.create<mlir::arith::ConstantIntOp>(loc, 42, 32);
    builder.create<mlir::func::ReturnOp>(loc, constantOp.getResult());
    
    // Run RelAlg→DB pass
    mlir::PassManager pm(context.get());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createLowerToDBPass());
    
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result)) << "RelAlg→DB pass failed on empty function";
    
    // Verify no DSA operations were created
    bool foundDSAOps = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
            foundDSAOps = true;
        }
    });
    
    EXPECT_FALSE(foundDSAOps) << "No DSA operations should be created for empty function";
}