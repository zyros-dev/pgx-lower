#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/DBToDSA/DBToDSA.h"

#include "execution/logging.h"

using namespace mlir;

class SimpleMaterializeTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("Setting up SimpleMaterializeTest");
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(SimpleMaterializeTest, DirectConversionTest) {
    PGX_DEBUG("Testing direct MaterializeOp conversion");
    
    // Create module and function
    auto moduleOp = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_direct", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a simple MaterializeOp directly
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    
    // Create a dummy value for the input
    auto dummyOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("dummy"),
        builder->getI64IntegerAttr(1)
    );
    
    // Add terminator to BaseTableOp body
    Block *body = &dummyOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(body);
    builder->create<::pgx::mlir::relalg::ReturnOp>(builder->getUnknownLoc());
    
    // Create MaterializeOp
    builder->setInsertionPointAfter(dummyOp);
    auto columnNames = builder->getArrayAttr({builder->getStringAttr("col1")});
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(),
        tableType,
        dummyOp.getResult(),
        columnNames
    );
    
    // Add function return
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Print module before conversion
    std::cout << "Module before conversion:\n";
    moduleOp.dump();
    
    // Apply just the MaterializeOp pattern
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<::pgx::mlir::relalg::RelAlgDialect>();
    target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
    
    mlir::pgx_conversion::DBToDSATypeConverter typeConverter;
    RewritePatternSet patterns(&context);
    patterns.add<mlir::pgx_conversion::MaterializeToDSAPattern>(typeConverter, &context);
    
    std::cout << "Applying conversion...\n";
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
        std::cout << "Conversion failed\n";
        FAIL() << "Conversion should succeed";
    }
    
    std::cout << "Module after conversion:\n";
    moduleOp.dump();
    
    // Check results
    bool foundMaterialize = false;
    bool foundCreateDS = false;
    bool foundFinalize = false;
    
    funcOp.walk([&](Operation *op) {
        if (isa<::pgx::mlir::relalg::MaterializeOp>(op))
            foundMaterialize = true;
        if (isa<::pgx::mlir::dsa::CreateDSOp>(op))
            foundCreateDS = true;
        if (isa<::pgx::mlir::dsa::FinalizeOp>(op))
            foundFinalize = true;
    });
    
    EXPECT_FALSE(foundMaterialize) << "MaterializeOp should be removed";
    EXPECT_TRUE(foundCreateDS) << "CreateDSOp should be created";
    EXPECT_TRUE(foundFinalize) << "FinalizeOp should be created";
}