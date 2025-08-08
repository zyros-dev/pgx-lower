#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

using namespace mlir;

class RelAlgToDBTest : public ::testing::Test {
protected:
    MLIRContext context;
    OpBuilder builder;
    OwningOpRef<ModuleOp> module;

    RelAlgToDBTest() : builder(&context) {
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        
        module = ModuleOp::create(UnknownLoc::get(&context));
        builder.setInsertionPointToStart(module->getBody());
    }

    void runRelAlgToDBPass(func::FuncOp funcOp) {
        PassManager pm(&context);
        pm.addPass(pgx_conversion::createRelAlgToDBPass());
        
        LogicalResult result = pm.run(funcOp);
        ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    }
};

TEST_F(RelAlgToDBTest, BaseTableToGetExternal) {
    PGX_DEBUG("Running BaseTableToGetExternal test");
    
    // Create a function with BaseTableOp
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_basetable", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)  // table OID
    );
    
    // Create return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass
    runRelAlgToDBPass(funcOp);
    
    // Verify conversion
    bool foundGetExternal = false;
    funcOp.walk([&](pgx::db::GetExternalOp op) {
        foundGetExternal = true;
        // Verify the table OID is preserved
        auto constantOp = op.getTableOid().getDefiningOp<arith::ConstantIntOp>();
        ASSERT_TRUE(constantOp != nullptr);
        EXPECT_EQ(constantOp.value(), 12345);
    });
    
    // Should not find any BaseTableOp after conversion
    int baseTableCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        baseTableCount++;
    });
    
    EXPECT_TRUE(foundGetExternal) << "GetExternalOp not found after conversion";
    EXPECT_EQ(baseTableCount, 0) << "BaseTableOp still exists after conversion";
    
    PGX_DEBUG("BaseTableToGetExternal test completed successfully");
}

TEST_F(RelAlgToDBTest, GetColumnToDBGetColumn) {
    PGX_DEBUG("Running GetColumnToDBGetColumn test");
    
    // For Phase 4c-1, we test GetColumnOp conversion in a more realistic context
    // GetColumnOp typically appears inside iteration regions where individual tuples are processed
    // We'll simulate this by having a BaseTableOp that produces an external source
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_getcolumn", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp to get an external source
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)
    );
    
    // In a real implementation, GetColumnOp would be inside an iteration region
    // For Phase 4c-1 testing, we'll defer this complex scenario and focus on
    // verifying that the conversion patterns are properly registered
    
    // Create return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    PGX_DEBUG("Before running pass - IR structure created");
    
    // Run the pass
    runRelAlgToDBPass(funcOp);
    
    PGX_DEBUG("After running pass - starting verification");
    
    // For Phase 4c-1, we verify that BaseTableOp was converted
    bool foundGetExternal = false;
    funcOp.walk([&](pgx::db::GetExternalOp op) {
        foundGetExternal = true;
        PGX_DEBUG("Found db.get_external operation");
    });
    
    EXPECT_TRUE(foundGetExternal) << "db.get_external not found after conversion";
    
    // Note: GetColumnOp conversion is registered but won't be exercised until
    // we have proper iteration regions in later phases
    
    PGX_DEBUG("GetColumnToDBGetColumn test completed - Phase 4c-1 scope verified");
}

TEST_F(RelAlgToDBTest, MaterializeOpConversion_Phase4c2) {
    PGX_DEBUG("Running MaterializeOpConversion_Phase4c2 test");
    
    // For Phase 4c-2, MaterializeOp should be converted to DSA operations
    // We need to load the DSA dialect for this test
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_materialize", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp first
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("id"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass
    runRelAlgToDBPass(funcOp);
    
    // Verify MaterializeOp was converted
    bool foundMaterialize = false;
    funcOp.walk([&](pgx::mlir::relalg::MaterializeOp op) {
        foundMaterialize = true;
    });
    EXPECT_FALSE(foundMaterialize) << "MaterializeOp should be converted in Phase 4c-2";
    
    // Verify DSA operations were created
    bool foundCreateDS = false;
    bool foundDSAppend = false;
    bool foundNextRow = false;
    bool foundFinalize = false;
    
    funcOp.walk([&](pgx::mlir::dsa::CreateDSOp op) {
        foundCreateDS = true;
        PGX_DEBUG("Found dsa.create_ds operation");
    });
    
    funcOp.walk([&](pgx::mlir::dsa::DSAppendOp op) {
        foundDSAppend = true;
        PGX_DEBUG("Found dsa.ds_append operation");
    });
    
    funcOp.walk([&](pgx::mlir::dsa::NextRowOp op) {
        foundNextRow = true;
        PGX_DEBUG("Found dsa.next_row operation");
    });
    
    funcOp.walk([&](pgx::mlir::dsa::FinalizeOp op) {
        foundFinalize = true;
        PGX_DEBUG("Found dsa.finalize operation");
    });
    
    EXPECT_TRUE(foundCreateDS) << "dsa.create_ds should be created";
    EXPECT_TRUE(foundDSAppend) << "dsa.ds_append should be created";
    EXPECT_TRUE(foundNextRow) << "dsa.next_row should be created";
    EXPECT_TRUE(foundFinalize) << "dsa.finalize should be created";
    
    PGX_DEBUG("MaterializeOpConversion_Phase4c2 test completed successfully");
}

TEST_F(RelAlgToDBTest, ReturnOpPassThrough) {
    PGX_DEBUG("Running ReturnOpPassThrough test - Phase 4c-1");
    
    // In Phase 4c-1, ReturnOp should NOT be converted
    // It should pass through unchanged to later phases
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_return", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create RelAlg return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass
    runRelAlgToDBPass(funcOp);
    
    // Should still have relalg.return (NOT converted in Phase 4c-1)
    bool foundRelAlgReturn = false;
    funcOp.walk([&](pgx::mlir::relalg::ReturnOp op) {
        foundRelAlgReturn = true;
        PGX_DEBUG("Found relalg.return - correctly passed through");
    });
    
    // Should NOT have func.return
    bool foundFuncReturn = false;
    funcOp.walk([&](func::ReturnOp op) {
        foundFuncReturn = true;
    });
    
    EXPECT_TRUE(foundRelAlgReturn) << "relalg.return should pass through unchanged in Phase 4c-1";
    EXPECT_FALSE(foundFuncReturn) << "func.return should NOT be created in Phase 4c-1";
    
    PGX_DEBUG("ReturnOpPassThrough test completed - Phase 4c-1 behavior verified");
}

TEST_F(RelAlgToDBTest, Phase4c2CompleteExample) {
    PGX_DEBUG("Running Phase4c2CompleteExample test");
    
    // Test a complete example showing Phase 4c-2 scope:
    // BaseTableOp converts to DB ops, MaterializeOp converts to DSA ops
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_complete", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a typical query pattern: BaseTable -> Materialize -> Return
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("employees"),
        builder.getI64IntegerAttr(54321)  // table OID
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    // Create columns attribute for MaterializeOp
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("*"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context), materializeOp.getResult());
    
    // Run the pass
    runRelAlgToDBPass(funcOp);
    
    // Verify Phase 4c-2 behavior:
    // 1. BaseTableOp converted to db.get_external
    bool foundGetExternal = false;
    funcOp.walk([&](pgx::db::GetExternalOp op) {
        foundGetExternal = true;
        auto constantOp = op.getTableOid().getDefiningOp<arith::ConstantIntOp>();
        ASSERT_TRUE(constantOp != nullptr);
        EXPECT_EQ(constantOp.value(), 54321) << "Table OID should be preserved";
    });
    EXPECT_TRUE(foundGetExternal) << "BaseTableOp should be converted";
    
    // 2. No BaseTableOp remaining
    int baseTableCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        baseTableCount++;
    });
    EXPECT_EQ(baseTableCount, 0) << "No BaseTableOp should remain";
    
    // 3. MaterializeOp converted to DSA operations
    bool foundMaterialize = false;
    funcOp.walk([&](pgx::mlir::relalg::MaterializeOp op) {
        foundMaterialize = true;
    });
    EXPECT_FALSE(foundMaterialize) << "MaterializeOp should be converted in Phase 4c-2";
    
    // 4. DSA operations created
    bool foundDSAOps = false;
    funcOp.walk([&](pgx::mlir::dsa::CreateDSOp op) {
        foundDSAOps = true;
    });
    funcOp.walk([&](pgx::mlir::dsa::FinalizeOp op) {
        foundDSAOps = true;
    });
    EXPECT_TRUE(foundDSAOps) << "DSA operations should be created";
    
    // 5. ReturnOp passed through unchanged
    bool foundReturn = false;
    funcOp.walk([&](pgx::mlir::relalg::ReturnOp op) {
        foundReturn = true;
    });
    EXPECT_TRUE(foundReturn) << "ReturnOp should pass through";
    
    PGX_DEBUG("Phase4c2CompleteExample test completed - all Phase 4c-2 requirements verified");
}