#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDSA/RelAlgToDSA.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class MaterializeOpLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(MaterializeOpLoweringTest, MaterializeOpLoweringCreatesProperDSAOperations) {
    // Create a simple MaterializeOp with a TupleStream input
    Location loc = builder->getUnknownLoc();
    
    // Create function as container
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_materialize", funcType);
    Block* funcBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(funcBlock);
    
    // Create a TupleStream type and value for MaterializeOp input
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto tupleStreamValue = funcBlock->addArgument(tupleStreamType, loc);
    
    // Create MaterializeOp
    auto columnsAttr = builder->getArrayAttr({});
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        loc, tableType, tupleStreamValue, columnsAttr);
    
    // Apply the lowering pattern
    mlir::pgx_conversion::MaterializeToResultBuilderPattern pattern(&context);
    PatternRewriter rewriter(&context);
    rewriter.setInsertionPoint(materializeOp);
    
    // Test that the pattern matches and rewrites successfully
    auto result = pattern.matchAndRewrite(materializeOp, rewriter);
    
    EXPECT_TRUE(succeeded(result));
    
    // Verify that DSA operations were created
    bool foundCreateDS = false;
    bool foundForOp = false;
    bool foundFinalizeOp = false;
    
    funcOp.walk([&](Operation* op) {
        if (auto createOp = dyn_cast<pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDS = true;
        }
        if (auto forOp = dyn_cast<pgx::mlir::dsa::ForOp>(op)) {
            foundForOp = true;
        }
        if (auto finalizeOp = dyn_cast<pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalizeOp = true;
        }
    });
    
    EXPECT_TRUE(foundCreateDS) << "MaterializeOp lowering should create CreateDSOp";
    EXPECT_TRUE(foundForOp) << "MaterializeOp lowering should create ForOp for nested iteration";  
    EXPECT_TRUE(foundFinalizeOp) << "MaterializeOp lowering should create FinalizeOp";
}

TEST_F(MaterializeOpLoweringTest, MaterializeOpLoweringCreatesNestedForOps) {
    // Create a simple MaterializeOp
    Location loc = builder->getUnknownLoc();
    
    // Create function as container
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_nested_for", funcType);
    Block* funcBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(funcBlock);
    
    // Create MaterializeOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto tupleStreamValue = funcBlock->addArgument(tupleStreamType, loc);
    auto columnsAttr = builder->getArrayAttr({});
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        loc, tableType, tupleStreamValue, columnsAttr);
    
    // Apply the lowering pattern
    mlir::pgx_conversion::MaterializeToResultBuilderPattern pattern(&context);
    PatternRewriter rewriter(&context);
    rewriter.setInsertionPoint(materializeOp);
    
    auto result = pattern.matchAndRewrite(materializeOp, rewriter);
    EXPECT_TRUE(succeeded(result));
    
    // Count the number of ForOps created (should be 2 for nested pattern)
    int forOpCount = 0;
    funcOp.walk([&](pgx::mlir::dsa::ForOp op) {
        forOpCount++;
    });
    
    EXPECT_EQ(forOpCount, 2) << "MaterializeOp lowering should create 2 nested ForOps (outer and inner)";
}

TEST_F(MaterializeOpLoweringTest, MaterializeOpLoweringHandlesTerminatorsCorrectly) {
    // Test that ForOps have proper implicit terminators
    Location loc = builder->getUnknownLoc();
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_terminators", funcType);
    Block* funcBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(funcBlock);
    
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto tupleStreamValue = funcBlock->addArgument(tupleStreamType, loc);
    auto columnsAttr = builder->getArrayAttr({});
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        loc, tableType, tupleStreamValue, columnsAttr);
    
    // Apply the lowering pattern
    mlir::pgx_conversion::MaterializeToResultBuilderPattern pattern(&context);
    PatternRewriter rewriter(&context);
    rewriter.setInsertionPoint(materializeOp);
    
    auto result = pattern.matchAndRewrite(materializeOp, rewriter);
    EXPECT_TRUE(succeeded(result));
    
    // Verify that ForOps have implicit YieldOp terminators
    funcOp.walk([&](pgx::mlir::dsa::ForOp forOp) {
        ASSERT_EQ(forOp.getBody().getBlocks().size(), 1U);
        Block& body = forOp.getBody().front();
        
        // ForOp should have SingleBlockImplicitTerminator trait
        // which means YieldOp terminators are automatically provided
        EXPECT_TRUE(body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>() || 
                   isa<pgx::mlir::dsa::YieldOp>(body.back())) 
            << "ForOp should have implicit YieldOp terminator";
    });
}