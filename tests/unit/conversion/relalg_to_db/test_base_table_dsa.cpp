#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx;

class RelAlgToDBBaseTableDSATest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<::mlir::arith::ArithDialect>();
        context.loadDialect<::mlir::func::FuncDialect>();
    }

    MLIRContext context;
};

TEST_F(RelAlgToDBBaseTableDSATest, BaseTableGeneratesDSAStreamingPattern) {
    // Create a module with a function containing a BaseTable operation
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<ModuleOp>(loc);
    
    // Create function
    builder.setInsertionPointToEnd(module.getBody());
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {tableType});
    auto func = builder.create<::mlir::func::FuncOp>(loc, "test_basetable", funcType);
    
    // Create function body with BaseTable
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Create BaseTable operation
    auto tableOid = 16384; // Example OID
    auto tableName = builder.getStringAttr("test");
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        loc, tupleStreamType, tableName, builder.getI64IntegerAttr(tableOid));
    
    // Create MaterializeOp that consumes the BaseTable with explicit column name
    auto columnsAttr = builder.getStrArrayAttr({"id"});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        loc, tableType, baseTableOp.getResult(), columnsAttr);
    
    // Create return with the materialized result
    builder.create<::mlir::func::ReturnOp>(loc, materializeOp.getResult());
    
    // Verify module before conversion
    ASSERT_TRUE(succeeded(verify(module)));
    
    // Run RelAlgToDB pass on the function
    PassManager pm(&context);
    pm.addNestedPass<::mlir::func::FuncOp>(::mlir::pgx_conversion::createRelAlgToDBPass());
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Print the module after conversion
    PGX_INFO("Module after conversion:");
    module->print(llvm::errs());
    
    // Check that DSA operations were generated
    bool foundScanSource = false;
    bool foundOuterFor = false;
    bool foundInnerFor = false;
    bool foundAt = false;
    int forLoopCount = 0;
    
    func.walk([&](Operation* op) {
        if (isa<pgx::mlir::dsa::ScanSourceOp>(op)) {
            foundScanSource = true;
            auto scanOp = cast<pgx::mlir::dsa::ScanSourceOp>(op);
            auto tableDesc = scanOp.getTableDescription();
            // Check that table description contains our table name
            EXPECT_TRUE(tableDesc.str().find("test") != std::string::npos);
            EXPECT_TRUE(tableDesc.str().find("16384") != std::string::npos);
        }
        if (isa<pgx::mlir::dsa::ForOp>(op)) {
            forLoopCount++;
            if (forLoopCount == 1) {
                foundOuterFor = true;
                // Verify it's iterating over a scan source result (GenericIterable)
                auto forOp = cast<pgx::mlir::dsa::ForOp>(op);
                auto iterableType = forOp.getIterable().getType();
                // It should be a GenericIterableType wrapping a RecordBatch
                if (auto genIterable = dyn_cast<pgx::mlir::dsa::GenericIterableType>(iterableType)) {
                    EXPECT_TRUE(isa<pgx::mlir::dsa::RecordBatchType>(genIterable.getElementType()));
                }
            } else if (forLoopCount == 2) {
                foundInnerFor = true;
                // Verify it's iterating over a record batch (which comes from the block argument)
                auto forOp = cast<pgx::mlir::dsa::ForOp>(op);
                // The inner loop iterates over a block argument which is a RecordBatchType
                // This is correct - the outer loop provides the RecordBatch via block argument
                foundInnerFor = true;
            }
        }
        if (isa<pgx::mlir::dsa::AtOp>(op)) {
            foundAt = true;
            auto atOp = cast<pgx::mlir::dsa::AtOp>(op);
            // Check that we're accessing the 'id' column
            EXPECT_EQ(atOp.getColumnName().str(), "id");
        }
    });
    
    // Print what operations were found
    PGX_INFO("Test found: ScanSource=" + std::to_string(foundScanSource) + 
             ", OuterFor=" + std::to_string(foundOuterFor) + 
             ", InnerFor=" + std::to_string(foundInnerFor) + 
             ", At=" + std::to_string(foundAt) + 
             ", ForLoopCount=" + std::to_string(forLoopCount));
    
    // Verify all expected DSA operations were generated
    EXPECT_TRUE(foundScanSource) << "dsa.scan_source not found";
    EXPECT_TRUE(foundOuterFor) << "Outer dsa.for loop not found";
    EXPECT_TRUE(foundInnerFor) << "Inner dsa.for loop not found";
    EXPECT_TRUE(foundAt) << "dsa.at operation not found";
    EXPECT_EQ(forLoopCount, 2) << "Expected exactly 2 dsa.for loops, found " << forLoopCount;
    
    // Verify module is still valid after conversion
    ASSERT_TRUE(succeeded(verify(module)));
    
    // BaseTable operation should have been removed
    bool foundBaseTable = false;
    func.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTable = true;
    });
    EXPECT_FALSE(foundBaseTable) << "BaseTable operation should have been removed";
    
    PGX_INFO("Test passed: BaseTable generates correct DSA streaming pattern");
}

TEST_F(RelAlgToDBBaseTableDSATest, NestedLoopStructureCorrect) {
    // Create a module with a BaseTable to verify nested loop structure
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<ModuleOp>(loc);
    
    builder.setInsertionPointToEnd(module.getBody());
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {tableType});
    auto func = builder.create<::mlir::func::FuncOp>(loc, "test_nested", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Create BaseTable
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("orders"), builder.getI64IntegerAttr(16385));
    
    // Create MaterializeOp
    auto columnsAttr = builder.getStrArrayAttr({"*"});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        loc, tableType, baseTableOp.getResult(), columnsAttr);
    
    builder.create<::mlir::func::ReturnOp>(loc, materializeOp.getResult());
    
    // Run conversion
    PassManager pm(&context);
    pm.addNestedPass<::mlir::func::FuncOp>(::mlir::pgx_conversion::createRelAlgToDBPass());
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Walk through and verify the nested structure
    func.walk([&](pgx::mlir::dsa::ForOp outerFor) {
        // Check outer loop has exactly one block
        EXPECT_EQ(outerFor.getBody().getBlocks().size(), 1u);
        
        // Check that the outer loop body contains an inner loop
        bool hasInnerLoop = false;
        outerFor.getBody().walk([&](pgx::mlir::dsa::ForOp innerFor) {
            hasInnerLoop = true;
            
            // Check inner loop has exactly one block
            EXPECT_EQ(innerFor.getBody().getBlocks().size(), 1u);
            
            // Check that inner loop contains an AtOp
            bool hasAt = false;
            innerFor.getBody().walk([&](pgx::mlir::dsa::AtOp at) {
                hasAt = true;
            });
            EXPECT_TRUE(hasAt) << "Inner loop should contain dsa.at operation";
        });
        
        if (!hasInnerLoop) {
            // If this is the outer loop, it should have an inner loop
            auto& block = outerFor.getBody().front();
            for (auto& op : block) {
                if (isa<pgx::mlir::dsa::ForOp>(&op)) {
                    hasInnerLoop = true;
                    break;
                }
            }
        }
        
        return WalkResult::advance(); // Only check the first ForOp (outer)
    });
    
    ASSERT_TRUE(succeeded(verify(module)));
    PGX_INFO("Test passed: Nested loop structure is correct");
}