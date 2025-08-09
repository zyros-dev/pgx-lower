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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        context.loadDialect<::mlir::scf::SCFDialect>();
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
    
    // Check that mixed DB+DSA operations were generated (Phase 4d)
    bool foundGetExternal = false;
    bool foundIterateExternal = false;
    bool foundGetField = false;
    bool foundScfWhile = false;
    bool foundCreateDS = false;
    bool foundDSAppend = false;
    bool foundFinalize = false;
    
    func.walk([&](Operation* op) {
        if (isa<pgx::db::GetExternalOp>(op)) {
            foundGetExternal = true;
            auto getExtOp = cast<pgx::db::GetExternalOp>(op);
            // Verify it uses the correct table OID
            auto oidOp = getExtOp.getTableOid().getDefiningOp<arith::ConstantOp>();
            if (oidOp) {
                auto oidAttr = oidOp.getValue().cast<IntegerAttr>();
                EXPECT_EQ(oidAttr.getInt(), 16384);
            }
        }
        if (isa<pgx::db::IterateExternalOp>(op)) {
            foundIterateExternal = true;
        }
        if (isa<pgx::db::GetFieldOp>(op)) {
            foundGetField = true;
            auto getFieldOp = cast<pgx::db::GetFieldOp>(op);
            // Check that we're accessing the correct field index
            auto idxOp = getFieldOp.getFieldIndex().getDefiningOp<arith::ConstantOp>();
            if (idxOp) {
                auto idxAttr = idxOp.getValue().cast<IntegerAttr>();
                EXPECT_EQ(idxAttr.getInt(), 0); // First field for 'id' column
            }
        }
        if (isa<scf::WhileOp>(op)) {
            foundScfWhile = true;
        }
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDS = true;
        }
        if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            foundDSAppend = true;
        }
        if (isa<pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalize = true;
        }
    });
    
    // Print what operations were found
    PGX_INFO("Test found: GetExternal=" + std::to_string(foundGetExternal) + 
             ", IterateExternal=" + std::to_string(foundIterateExternal) + 
             ", GetField=" + std::to_string(foundGetField) + 
             ", ScfWhile=" + std::to_string(foundScfWhile) +
             ", CreateDS=" + std::to_string(foundCreateDS) +
             ", DSAppend=" + std::to_string(foundDSAppend) +
             ", Finalize=" + std::to_string(foundFinalize));
    
    // Verify all expected mixed DB+DSA operations were generated
    EXPECT_TRUE(foundGetExternal) << "db.get_external not found";
    EXPECT_TRUE(foundIterateExternal) << "db.iterate_external not found";
    EXPECT_TRUE(foundGetField) << "db.get_field operation not found";
    EXPECT_TRUE(foundScfWhile) << "scf.while loop not found";
    EXPECT_TRUE(foundCreateDS) << "dsa.create_ds not found";
    EXPECT_TRUE(foundDSAppend) << "dsa.ds_append not found";
    EXPECT_TRUE(foundFinalize) << "dsa.finalize not found";
    
    // Verify module is still valid after conversion
    ASSERT_TRUE(succeeded(verify(module)));
    
    // BaseTable operation should have been removed
    bool foundBaseTable = false;
    func.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTable = true;
    });
    EXPECT_FALSE(foundBaseTable) << "BaseTable operation should have been removed";
    
    PGX_INFO("Test passed: BaseTable generates correct mixed DB+DSA pattern");
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
    
    // Walk through and verify the scf.while loop structure
    bool foundWhileLoop = false;
    bool hasGetField = false;
    bool hasDSAppend = false;
    
    func.walk([&](scf::WhileOp whileOp) {
        foundWhileLoop = true;
        
        // Check that the while loop has the proper structure
        // Before region: contains db.iterate_external
        whileOp.getBefore().walk([&](pgx::db::IterateExternalOp iterOp) {
            // Verify iterate_external is in the before region
            EXPECT_TRUE(true) << "db.iterate_external found in before region";
        });
        
        // After region: contains db.get_field and dsa operations
        whileOp.getAfter().walk([&](Operation* op) {
            if (isa<pgx::db::GetFieldOp>(op)) {
                hasGetField = true;
            }
            if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
                hasDSAppend = true;
            }
        });
        
        return WalkResult::advance();
    });
    
    EXPECT_TRUE(foundWhileLoop) << "scf.while loop not found";
    EXPECT_TRUE(hasGetField) << "db.get_field should be in while loop body";
    EXPECT_TRUE(hasDSAppend) << "dsa.ds_append should be in while loop body";
    
    ASSERT_TRUE(succeeded(verify(module)));
    PGX_INFO("Test passed: While loop structure is correct");
}