#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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

// Test to verify MaterializeOp generates the correct DB operation sequence
TEST(MaterializeDBOpsTest, VerifyDBSequence) {
    PGX_DEBUG("Testing MaterializeOp → DB operation sequence generation");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
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
    
    // Verify the DB operation sequence
    // Should have: get_external → iterate_external → get_field → store_result → stream_results
    int getExternalCount = 0;
    int iterateExternalCount = 0;
    int getFieldCount = 0;
    int storeResultCount = 0;
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
        else if (isa<pgx::db::StoreResultOp>(op)) {
            storeResultCount++;
            PGX_DEBUG("Found db.store_result");
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
    
    // Verify the exact sequence
    EXPECT_EQ(getExternalCount, 1) << "Should have exactly one db.get_external";
    EXPECT_EQ(iterateExternalCount, 1) << "Should have exactly one db.iterate_external";
    EXPECT_EQ(getFieldCount, 1) << "Should have exactly one db.get_field";
    EXPECT_EQ(storeResultCount, 1) << "Should have exactly one db.store_result";
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
    
    PGX_DEBUG("MaterializeOp → DB operation sequence verification completed");
}

// Test to verify DB operations only (no DSA operations)
TEST(MaterializeDBOpsTest, NoMixedOperations) {
    PGX_DEBUG("Testing RelAlgToDB generates only DB operations (no DSA)");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_no_dsa", funcType);
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
    
    // Verify we have ONLY DB operations, NO DSA operations
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
            PGX_ERROR("UNEXPECTED: Found DSA operation: " + op->getName().getStringRef().str());
        }
    });
    
    EXPECT_TRUE(hasDBOps) << "Should have DB operations from RelAlg conversion";
    EXPECT_FALSE(hasDSAOps) << "Should NOT have DSA operations - RelAlgToDB should generate only DB ops";
    
    PGX_DEBUG("No DSA operations test completed - only DB operations found");
}

// Test pass infrastructure
TEST(MaterializeDBOpsTest, PassExists) {
    PGX_DEBUG("Running PassExists test - verifying pass can be created");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
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