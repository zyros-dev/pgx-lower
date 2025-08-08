// Test suite for Phase 4c-3: MaterializeTranslator DSA-based Result Materialization
// In Phase 4c-3, the MaterializeTranslator uses DSA operations (create_ds, ds_append, 
// next_row, finalize) to materialize query results following the LingoDB architecture.
// The RelAlgToDB pass now generates mixed DB+DSA operations as per the design document.

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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "execution/logging.h"

using namespace mlir;

// Test to verify MaterializeOp generates the correct DB+DSA operation sequence for Phase 4c-3
TEST(MaterializeDBOpsTest, VerifyDBSequence) {
    PGX_DEBUG("Testing MaterializeOp → DB+DSA operation sequence generation (Phase 4c-3)");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_materialize_db", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test"),
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
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the DB+DSA operation sequence for Phase 4c-3
    // BaseTable generates: get_external → iterate_external → get_field  
    // Materialize generates: create_ds → ds_append → next_row → finalize → stream_results
    int getExternalCount = 0;
    int iterateExternalCount = 0;
    int getFieldCount = 0;
    int createDsCount = 0;
    int dsAppendCount = 0;
    int nextRowCount = 0;
    int finalizeCount = 0;
    int streamResultsCount = 0;
    int scfWhileCount = 0;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::db::GetExternalOp>(op)) {
            getExternalCount++;
            PGX_DEBUG("Found db.get_external");
        }
        else if (isa<pgx::db::IterateExternalOp>(op)) {
            iterateExternalCount++;
            PGX_DEBUG("Found db.iterate_external");
        }
        else if (isa<pgx::db::GetFieldOp>(op)) {
            getFieldCount++;
            PGX_DEBUG("Found db.get_field");
        }
        else if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            createDsCount++;
            PGX_DEBUG("Found dsa.create_ds");
        }
        else if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            dsAppendCount++;
            PGX_DEBUG("Found dsa.ds_append");
        }
        else if (isa<pgx::mlir::dsa::NextRowOp>(op)) {
            nextRowCount++;
            PGX_DEBUG("Found dsa.next_row");
        }
        else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) {
            finalizeCount++;
            PGX_DEBUG("Found dsa.finalize");
        }
        else if (isa<pgx::db::StreamResultsOp>(op)) {
            streamResultsCount++;
            PGX_DEBUG("Found db.stream_results");
        }
        else if (isa<scf::WhileOp>(op)) {
            scfWhileCount++;
            PGX_DEBUG("Found scf.while loop");
        }
    });
    
    // Verify the exact sequence for Phase 4c-3
    EXPECT_EQ(getExternalCount, 1) << "Should have exactly one db.get_external";
    EXPECT_EQ(iterateExternalCount, 1) << "Should have exactly one db.iterate_external";
    EXPECT_EQ(getFieldCount, 1) << "Should have exactly one db.get_field";
    EXPECT_EQ(createDsCount, 1) << "Should have exactly one dsa.create_ds";
    EXPECT_GE(dsAppendCount, 1) << "Should have at least one dsa.ds_append (one per consume call)";
    EXPECT_GE(nextRowCount, 1) << "Should have at least one dsa.next_row (one per consume call)";
    EXPECT_EQ(finalizeCount, 1) << "Should have exactly one dsa.finalize";
    EXPECT_EQ(streamResultsCount, 1) << "Should have exactly one db.stream_results";
    EXPECT_EQ(scfWhileCount, 1) << "Should have exactly one scf.while loop";
    
    // Verify MaterializeOp was removed
    int materializeCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::MaterializeOp op) {
        materializeCount++;
    });
    EXPECT_EQ(materializeCount, 0) << "MaterializeOp should be converted";
    
    // Verify BaseTableOp was removed
    int baseTableCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        baseTableCount++;
    });
    EXPECT_EQ(baseTableCount, 0) << "BaseTableOp should be converted";
    
    PGX_DEBUG("MaterializeOp → DB+DSA operation sequence verification completed");
}

// Test to verify RelAlgToDB generates mixed DB+DSA operations for Phase 4c-3
TEST(MaterializeDBOpsTest, VerifyMixedDBDSAOperations) {
    PGX_DEBUG("Testing RelAlgToDB generates mixed DB+DSA operations (Phase 4c-3)");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_mixed_ops", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("employees"),
        builder.getI64IntegerAttr(54321)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("*"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify we have BOTH DB and DSA operations (Phase 4c-3 architecture)
    bool hasDBOps = false;
    bool hasDSAOps = false;
    
    funcOp.walk([&](Operation *op) {
        auto dialectNamespace = op->getDialect() ? op->getDialect()->getNamespace() : "";
        if (dialectNamespace == "db") {
            hasDBOps = true;
            PGX_DEBUG("Found DB operation: " + op->getName().getStringRef().str());
        }
        if (dialectNamespace == "dsa") {
            hasDSAOps = true;
            PGX_DEBUG("Found DSA operation: " + op->getName().getStringRef().str());
        }
    });
    
    EXPECT_TRUE(hasDBOps) << "Should have DB operations from BaseTable conversion";
    EXPECT_TRUE(hasDSAOps) << "Should have DSA operations from Materialize conversion (Phase 4c-3)";
    
    PGX_DEBUG("Mixed DB+DSA operations test completed - Phase 4c-3 architecture verified");
}

// Test pass infrastructure
TEST(MaterializeDBOpsTest, PassExists) {
    PGX_DEBUG("Running PassExists test - verifying pass can be created");
    
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
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_pass_exists", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple return
    builder.create<func::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass - it should succeed even as a no-op
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed as no-op";
    
    PGX_DEBUG("PassExists test completed - pass infrastructure is working");
}