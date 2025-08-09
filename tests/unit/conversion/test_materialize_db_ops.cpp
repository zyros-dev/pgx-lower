// Test suite for Phase 4d-1: MaterializeTranslator DSA-based Result Materialization
// RESTORED: DSA operations restored based on reviewer feedback

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
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Test fixture for proper MLIR context lifecycle management
class MaterializeDBOpsTest : public ::testing::Test {
protected:
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    
    void SetUp() override {
        // Load required dialects
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<scf::SCFDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    void TearDown() override {
        // Ensure proper cleanup
        builder.reset();
    }
};

// Test that MaterializeOp generates hybrid DSA + PostgreSQL SPI operations
TEST_F(MaterializeDBOpsTest, MaterializeGeneratesHybridOps) {
    PGX_DEBUG("Testing MaterializeOp generates hybrid DSA + PostgreSQL SPI operations");
    
    // Create module and function
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    
    // Create function with expected signature after conversion
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_materialize_hybrid", funcType);
    auto entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto tableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345)  // table_oid
    );
    
    // Create MaterializeOp  
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto columnsAttr = builder->getArrayAttr({builder->getStringAttr("id")});
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        loc,
        tableType,
        tableOp.getResult(),
        columnsAttr
    );
    
    // Add return to function
    builder->create<func::ReturnOp>(loc);
    
    // Apply RelAlgToDB conversion
    PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    if (failed(pm.run(func))) {
        FAIL() << "RelAlgToDB pass failed";
    }
    
    // Verify hybrid DSA + PostgreSQL SPI operations were generated
    bool foundCreateDS = false;
    bool foundDSAppend = false;
    bool foundNextRow = false;
    bool foundStoreResult = false;
    bool foundStreamResults = false;
    
    module.walk([&](Operation* op) {
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDS = true;
            // Verify it creates a TableBuilder
            auto createOp = cast<pgx::mlir::dsa::CreateDSOp>(op);
            EXPECT_TRUE(createOp.getDs().getType().isa<pgx::mlir::dsa::TableBuilderType>());
        } else if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            foundDSAppend = true;
        } else if (isa<pgx::mlir::dsa::NextRowOp>(op)) {
            foundNextRow = true;
        } else if (isa<pgx::db::StoreResultOp>(op)) {
            foundStoreResult = true;
        } else if (isa<pgx::db::StreamResultsOp>(op)) {
            foundStreamResults = true;
        }
    });
    
    // Phase 4d-6: Updated expectations for hybrid architecture
    EXPECT_TRUE(foundCreateDS) << "Should generate dsa.create_ds for internal table building";
    EXPECT_TRUE(foundDSAppend) << "Should generate dsa.ds_append for column values";
    EXPECT_TRUE(foundNextRow) << "Should generate dsa.next_row to finalize rows";
    EXPECT_TRUE(foundStoreResult) << "Should generate db.store_result for PostgreSQL output";
    EXPECT_TRUE(foundStreamResults) << "Should generate db.stream_results for result streaming";
    
    // Verify NO dsa.finalize in hybrid architecture
    bool foundFinalize = false;
    module.walk([&](pgx::mlir::dsa::FinalizeOp op) {
        foundFinalize = true;
    });
    EXPECT_FALSE(foundFinalize) << "Should NOT generate dsa.finalize in hybrid architecture";
    
    PGX_DEBUG("MaterializeOp correctly generates hybrid operations");
}

// Test MaterializeOp with nullable types
// Removed test for nullable types - MaterializeOp doesn't handle nullable types directly anymore
// The DSA operations handle the type conversions internally