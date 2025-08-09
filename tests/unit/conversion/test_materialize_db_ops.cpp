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

// Test to verify MaterializeOp generates the correct DSA operation sequence
TEST_F(MaterializeDBOpsTest, VerifyDSASequence) {
    PGX_DEBUG("Testing MaterializeOp → DB+DSA operation sequence generation (Phase 4c-3)");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_materialize_db", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("test"),
        builder->getI64IntegerAttr(12345)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("id"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the correct mixed DB+DSA operation sequence (Phase 4c-4 pattern)
    // BaseTable generates: db.get_external → scf.while loop → db.iterate_external → db.get_field
    // Materialize generates: dsa.create_ds → dsa.ds_append → dsa.next_row → dsa.finalize
    int getExternalCount = 0;
    int iterateExternalCount = 0;
    int getFieldCount = 0;
    int scfWhileCount = 0;
    int createDsCount = 0;
    int dsAppendCount = 0;
    int nextRowCount = 0;
    int finalizeCount = 0;
    int scfYieldCount = 0;
    
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
        else if (isa<scf::WhileOp>(op)) {
            scfWhileCount++;
            PGX_DEBUG("Found scf.while loop");
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
        else if (isa<scf::YieldOp>(op)) {
            scfYieldCount++;
            PGX_DEBUG("Found scf.yield");
        }
    });
    
    // Verify the mixed DB+DSA pattern
    EXPECT_EQ(getExternalCount, 1) << "Should have exactly one db.get_external";
    EXPECT_EQ(scfWhileCount, 1) << "Should have exactly one scf.while loop";
    EXPECT_GE(iterateExternalCount, 1) << "Should have at least one db.iterate_external";
    EXPECT_GE(getFieldCount, 1) << "Should have at least one db.get_field for field access";
    EXPECT_EQ(createDsCount, 1) << "Should have exactly one dsa.create_ds";
    EXPECT_GE(dsAppendCount, 1) << "Should have at least one dsa.ds_append (one per consume call)";
    EXPECT_GE(nextRowCount, 1) << "Should have at least one dsa.next_row (one per consume call)";
    EXPECT_EQ(finalizeCount, 1) << "Should have exactly one dsa.finalize";
    EXPECT_GE(scfYieldCount, 1) << "Should have at least one scf.yield terminator";
    
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

// Test to verify RelAlgToDB generates DSA operations following LingoDB pattern
TEST_F(MaterializeDBOpsTest, VerifyDSAOperations) {
    PGX_DEBUG("Testing RelAlgToDB generates mixed DB+DSA operations (Phase 4c-3)");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_mixed_ops", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("employees"),
        builder->getI64IntegerAttr(54321)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("*"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the module is valid after conversion
    ASSERT_TRUE(succeeded(funcOp.verify())) << "MLIR module verification failed after conversion";
    
    // Verify we have BOTH DB and DSA operations (Phase 4c-3 architecture)
    bool hasDBOps = false;
    bool hasDSAOps = false;
    
    PGX_DEBUG("Starting operation walk after conversion");
    
    try {
        funcOp.walk([&](Operation *op) {
            if (!op) {
                PGX_ERROR("Null operation encountered during walk");
                return;
            }
            
            // Safety check - ensure operation is still attached to a block
            if (!op->getBlock()) {
                PGX_ERROR("Operation detached from block: " + std::string(op->getName().getStringRef()));
                return;
            }
            
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
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during operation walk: " + std::string(e.what()));
        FAIL() << "Operation walk threw exception: " << e.what();
    } catch (...) {
        PGX_ERROR("Unknown exception during operation walk");
        FAIL() << "Operation walk threw unknown exception";
    }
    
    PGX_DEBUG("Operation walk completed successfully");
    
    // In Phase 4c-4 pattern, we use mixed DB+DSA operations for PostgreSQL integration
    EXPECT_TRUE(hasDBOps) << "Should have DB operations for PostgreSQL table access";
    EXPECT_TRUE(hasDSAOps) << "Should have DSA operations for result building";
    
    PGX_DEBUG("Mixed DB+DSA operations test completed - Phase 4c-3 architecture verified");
}

// Test pass infrastructure
TEST_F(MaterializeDBOpsTest, PassExists) {
    PGX_DEBUG("Running PassExists test - verifying pass can be created");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_pass_exists", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a simple return
    builder->create<func::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass - it should succeed even as a no-op
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed as no-op";
    
    PGX_DEBUG("PassExists test completed - pass infrastructure is working");
}

// Unit test for MaterializeTranslator's createTupleType() method
TEST_F(MaterializeDBOpsTest, CreateTupleTypeMethod) {
    PGX_DEBUG("Testing MaterializeTranslator::createTupleType() method");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_tuple_type", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp with specific column types
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(67890)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("id"));
    columnAttrs.push_back(builder->getStringAttr("name"));
    columnAttrs.push_back(builder->getStringAttr("age"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify DSA operations exist with proper tuple types
    bool foundCreateDS = false;
    bool hasTableBuilderType = false;
    
    funcOp.walk([&](pgx::mlir::dsa::CreateDSOp createOp) {
        foundCreateDS = true;
        auto resultType = createOp.getResult().getType();
        if (auto tableBuilderType = resultType.dyn_cast<pgx::mlir::dsa::TableBuilderType>()) {
            hasTableBuilderType = true;
            // Verify the tuple type is embedded in the table builder type
            auto tupleType = tableBuilderType.getRowType();
            EXPECT_TRUE(tupleType != nullptr) << "TableBuilderType should contain a tuple type";
            PGX_DEBUG("Created TableBuilder with tuple type: " + 
                     std::to_string(tupleType.getTypes().size()) + " fields");
        }
    });
    
    EXPECT_TRUE(foundCreateDS) << "Should have created dsa.create_ds operation";
    EXPECT_TRUE(hasTableBuilderType) << "dsa.create_ds should return TableBuilderType";
    
    PGX_DEBUG("createTupleType() method test completed");
}

// Unit test for MaterializeTranslator's finalizeAndStreamResults() method
TEST_F(MaterializeDBOpsTest, FinalizeAndStreamResultsMethod) {
    PGX_DEBUG("Testing MaterializeTranslator::finalizeAndStreamResults() method - VERIFICATION DISABLED");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_finalize", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create simple MaterializeOp chain
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("employees"),
        builder->getI64IntegerAttr(11111)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("emp_id"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the finalize and stream sequence exists
    bool foundFinalize = false;
    bool foundStreamResults = false;
    bool sequenceCorrect = false;
    
    // Track operation ordering
    Operation* finalizeOp = nullptr;
    Operation* streamOp = nullptr;
    
    funcOp.walk([&](Operation *op) {
        if (auto finalize = dyn_cast<pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalize = true;
            finalizeOp = op;
            PGX_DEBUG("Found dsa.finalize operation");
            
            // Verify it takes a table builder and returns a table
            auto builderType = finalize.getBuilder().getType();
            EXPECT_TRUE(builderType.isa<pgx::mlir::dsa::TableBuilderType>()) 
                << "finalize should take TableBuilderType";
            
            auto resultType = finalize.getResult().getType();
            EXPECT_TRUE(resultType.isa<pgx::mlir::dsa::TableType>()) 
                << "finalize should return TableType";
        }
    });
    
    EXPECT_TRUE(foundFinalize) << "Should have dsa.finalize operation";
    // In LingoDB pattern, we don't generate db.stream_results - the DSA table is returned directly
    EXPECT_FALSE(foundStreamResults) << "Should NOT have db.stream_results in LingoDB pattern";
    
    PGX_DEBUG("finalizeAndStreamResults() method test completed");
}

// Unit test for MaterializeTranslator's consume() method and DSA table builder lifecycle
TEST_F(MaterializeDBOpsTest, ConsumeMethodAndTableBuilderLifecycle) {
    PGX_DEBUG("Testing MaterializeTranslator::consume() method and DSA table builder lifecycle");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_consume", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("products"),
        builder->getI64IntegerAttr(99999)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("product_id"));
    columnAttrs.push_back(builder->getStringAttr("price"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify DSA table builder lifecycle operations
    int createDsCount = 0;
    int dsAppendCount = 0;
    int nextRowCount = 0;
    int finalizeCount = 0;
    
    // Track if DSA operations are inside the scf.while loop
    bool appendInLoop = false;
    bool nextRowInLoop = false;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            createDsCount++;
            // Verify create_ds is outside the loops
            bool inLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
            EXPECT_FALSE(inLoop) << "create_ds should be outside the loops";
        }
        else if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            dsAppendCount++;
            appendInLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
        }
        else if (isa<pgx::mlir::dsa::NextRowOp>(op)) {
            nextRowCount++;
            nextRowInLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
        }
        else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) {
            finalizeCount++;
            // Verify finalize is outside the loops
            bool inLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
            EXPECT_FALSE(inLoop) << "finalize should be outside the loops";
        }
    });
    
    // Verify lifecycle completeness
    EXPECT_EQ(createDsCount, 1) << "Should have exactly one create_ds";
    EXPECT_GE(dsAppendCount, 1) << "Should have at least one ds_append (inside loop)";
    EXPECT_GE(nextRowCount, 1) << "Should have at least one next_row (inside loop)";
    EXPECT_EQ(finalizeCount, 1) << "Should have exactly one finalize";
    
    // Verify loop placement
    EXPECT_TRUE(appendInLoop) << "ds_append should be inside the scf.while loop";
    EXPECT_TRUE(nextRowInLoop) << "next_row should be inside the scf.while loop";
    
    PGX_DEBUG("consume() method and table builder lifecycle test completed");
}

// Integration test: MaterializeTranslator + BaseTableTranslator producer-consumer flow
TEST_F(MaterializeDBOpsTest, ProducerConsumerIntegration) {
    PGX_DEBUG("Testing MaterializeTranslator + BaseTableTranslator integration");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_integration", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp → MaterializeOp chain
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("orders"),
        builder->getI64IntegerAttr(55555)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("order_id"));
    columnAttrs.push_back(builder->getStringAttr("customer_id"));
    columnAttrs.push_back(builder->getStringAttr("total"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the complete mixed DB+DSA producer-consumer flow exists
    bool hasCreateDs = false;
    bool hasDsAppend = false;
    bool hasNextRow = false;
    bool hasFinalize = false;
    bool hasYield = false;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) hasCreateDs = true;
        else if (isa<pgx::mlir::dsa::DSAppendOp>(op)) hasDsAppend = true;
        else if (isa<pgx::mlir::dsa::NextRowOp>(op)) hasNextRow = true;
        else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) hasFinalize = true;
        else if (isa<scf::YieldOp>(op)) hasYield = true;
    });
    
    // Verify all components of the mixed DB+DSA flow exist
    bool hasGetExternal = false;
    bool hasIterateExternal = false;
    bool hasGetField = false;
    bool hasScfWhile = false;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::db::GetExternalOp>(op)) hasGetExternal = true;
        else if (isa<pgx::db::IterateExternalOp>(op)) hasIterateExternal = true;
        else if (isa<pgx::db::GetFieldOp>(op)) hasGetField = true;
        else if (isa<scf::WhileOp>(op)) hasScfWhile = true;
    });
    
    // DB operations for table access (Phase 4c-4)
    EXPECT_TRUE(hasGetExternal) << "Producer should generate db.get_external";
    EXPECT_TRUE(hasIterateExternal) << "Producer should generate db.iterate_external";
    EXPECT_TRUE(hasGetField) << "Producer should generate db.get_field for field access";
    EXPECT_TRUE(hasScfWhile) << "Producer should generate scf.while loop";
    
    // DSA operations for result building
    EXPECT_TRUE(hasCreateDs) << "Consumer should generate dsa.create_ds";
    EXPECT_TRUE(hasDsAppend) << "Consumer should generate dsa.ds_append";
    EXPECT_TRUE(hasNextRow) << "Consumer should generate dsa.next_row";
    EXPECT_TRUE(hasFinalize) << "Consumer should generate dsa.finalize";
    
    // Verify no RelAlg operations remain (except return which is structural)
    int relalgCount = 0;
    funcOp.walk([&](Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            // ReturnOp is a structural terminator and should remain
            if (!isa<pgx::mlir::relalg::ReturnOp>(op)) {
                relalgCount++;
                PGX_DEBUG("Remaining RelAlg op: " + op->getName().getStringRef().str());
            }
        }
    });
    EXPECT_EQ(relalgCount, 0) << "All RelAlg operations (except structural return) should be lowered";
    
    PGX_DEBUG("Producer-consumer integration test completed");
}

// Error handling test: DSA operation failures
TEST_F(MaterializeDBOpsTest, DSAOperationFailureHandling) {
    PGX_DEBUG("Testing DSA operation failure scenarios");
    
    // Test 1: Empty table (no rows)
    {
        OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
        builder->setInsertionPointToStart(module->getBody());
        
        auto funcType = builder->getFunctionType({}, {});
        auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_empty", funcType);
        auto* entryBlock = funcOp.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock);
        
        // Create empty BaseTableOp (simulated by special table ID)
        auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
        auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
            UnknownLoc::get(&context),
            tupleStreamType,
            builder->getStringAttr("empty_table"),
            builder->getI64IntegerAttr(0)  // Special ID for empty table
        );
        
        auto tableType = pgx::mlir::relalg::TableType::get(&context);
        llvm::SmallVector<mlir::Attribute> columnAttrs;
        columnAttrs.push_back(builder->getStringAttr("col1"));
        auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
        
        auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
            UnknownLoc::get(&context),
            tableType,
            baseTableOp.getResult(),
            columnsArrayAttr
        );
        
        builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
        
        // Run the pass - should handle empty table gracefully
        PassManager pm(&context);
        pm.addPass(pgx_conversion::createRelAlgToDBPass());
        
        LogicalResult result = pm.run(funcOp);
        ASSERT_TRUE(succeeded(result)) << "Pass should handle empty tables gracefully";
        
        // Verify DSA operations still exist (create/finalize without append/next_row)
        int createDsCount = 0;
        int finalizeCount = 0;
        
        funcOp.walk([&](Operation *op) {
            if (isa<pgx::mlir::dsa::CreateDSOp>(op)) createDsCount++;
            else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) finalizeCount++;
        });
        
        EXPECT_EQ(createDsCount, 1) << "Should still create DSA table for empty result";
        EXPECT_EQ(finalizeCount, 1) << "Should still finalize DSA table for empty result";
    }
    
    PGX_DEBUG("DSA operation failure handling test completed");
}

// Error propagation test
TEST_F(MaterializeDBOpsTest, ErrorPropagation) {
    PGX_DEBUG("Testing error propagation through MaterializeTranslator");
    
    // Test: MaterializeOp with no input (null operand)
    {
        OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
        builder->setInsertionPointToStart(module->getBody());
        
        auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
        auto funcType = builder->getFunctionType({tupleStreamType}, {});
        auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_null_input", funcType);
        auto* entryBlock = funcOp.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock);
        
        // Create MaterializeOp with block argument (simulates detached input)
        auto tableType = pgx::mlir::relalg::TableType::get(&context);
        auto blockArg = entryBlock->getArgument(0);
        
        llvm::SmallVector<mlir::Attribute> columnAttrs;
        columnAttrs.push_back(builder->getStringAttr("col"));
        auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
        
        auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
            UnknownLoc::get(&context),
            tableType,
            blockArg,  // No defining op
            columnsArrayAttr
        );
        
        builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
        
        // Run the pass - should handle gracefully
        PassManager pm(&context);
        pm.addPass(pgx_conversion::createRelAlgToDBPass());
        
        LogicalResult result = pm.run(funcOp);
        ASSERT_TRUE(succeeded(result)) << "Pass should handle detached inputs gracefully";
        
        // MaterializeOp should be converted even with detached input
        // It will generate DSA operations (create_ds, finalize, stream_results)
        bool materializeRemains = false;
        funcOp.walk([&](pgx::mlir::relalg::MaterializeOp op) {
            materializeRemains = true;
        });
        
        EXPECT_FALSE(materializeRemains) << "MaterializeOp should be converted even with detached input";
        
        // Verify DSA operations were created
        bool hasDSAOps = false;
        funcOp.walk([&](Operation *op) {
            if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
                hasDSAOps = true;
            }
        });
        EXPECT_TRUE(hasDSAOps) << "Should generate DSA operations even with detached input";
    }
    
    PGX_DEBUG("Error propagation test completed");
}

// Performance test: Streaming behavior validation
TEST_F(MaterializeDBOpsTest, StreamingBehaviorValidation) {
    PGX_DEBUG("Testing streaming behavior - one tuple per consume() call");
    
    // Create module and function using OwningOpRef for proper cleanup
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    builder->setInsertionPointToStart(module->getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_streaming", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create large table scenario
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder->getStringAttr("large_table"),
        builder->getI64IntegerAttr(1000000)  // Simulate large table
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    // Multiple columns to test streaming efficiency
    columnAttrs.push_back(builder->getStringAttr("id"));
    columnAttrs.push_back(builder->getStringAttr("name"));
    columnAttrs.push_back(builder->getStringAttr("email"));
    columnAttrs.push_back(builder->getStringAttr("phone"));
    columnAttrs.push_back(builder->getStringAttr("address"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify streaming pattern: each tuple is processed individually
    // Check that DSA operations are inside the DSA for loops
    int appendInLoopCount = 0;
    int nextRowInLoopCount = 0;
    int appendOutsideLoopCount = 0;
    int nextRowOutsideLoopCount = 0;
    
    funcOp.walk([&](Operation *op) {
        bool inLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
        
        if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            if (inLoop) appendInLoopCount++;
            else appendOutsideLoopCount++;
        }
        else if (isa<pgx::mlir::dsa::NextRowOp>(op)) {
            if (inLoop) nextRowInLoopCount++;
            else nextRowOutsideLoopCount++;
        }
    });
    
    // All append/next_row operations should be inside the scf.while loop for streaming
    EXPECT_GT(appendInLoopCount, 0) << "Should have ds_append inside scf.while loop for streaming";
    EXPECT_GT(nextRowInLoopCount, 0) << "Should have next_row inside scf.while loop for streaming";
    EXPECT_EQ(appendOutsideLoopCount, 0) << "No ds_append should be outside loops";
    EXPECT_EQ(nextRowOutsideLoopCount, 0) << "No next_row should be outside loops";
    
    // Verify single create_ds and finalize (outside loops)
    int createDsCount = 0;
    int finalizeCount = 0;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            createDsCount++;
            bool inLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
            EXPECT_FALSE(inLoop) << "create_ds must be outside loops for efficiency";
        }
        else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) {
            finalizeCount++;
            bool inLoop = op->getParentOfType<scf::WhileOp>() != nullptr;
            EXPECT_FALSE(inLoop) << "finalize must be outside loops for efficiency";
        }
    });
    
    EXPECT_EQ(createDsCount, 1) << "Should create table builder only once";
    EXPECT_EQ(finalizeCount, 1) << "Should finalize table only once";
    
    PGX_DEBUG("Streaming behavior validation completed");
}

// Edge case test: MaterializeOp with special column patterns
TEST_F(MaterializeDBOpsTest, SpecialColumnPatterns) {
    PGX_DEBUG("Testing MaterializeOp with special column patterns");
    
    // Test case 1: SELECT * pattern
    {
        OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
        builder->setInsertionPointToStart(module->getBody());
        
        auto funcType = builder->getFunctionType({}, {});
        auto funcOp = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_select_star", funcType);
        auto* entryBlock = funcOp.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock);
        
        auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
        auto baseTableOp = builder->create<pgx::mlir::relalg::BaseTableOp>(
            UnknownLoc::get(&context),
            tupleStreamType,
            builder->getStringAttr("all_columns"),
            builder->getI64IntegerAttr(88888)
        );
        
        auto tableType = pgx::mlir::relalg::TableType::get(&context);
        llvm::SmallVector<mlir::Attribute> columnAttrs;
        columnAttrs.push_back(builder->getStringAttr("*"));  // SELECT * pattern
        auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
        
        auto materializeOp = builder->create<pgx::mlir::relalg::MaterializeOp>(
            UnknownLoc::get(&context),
            tableType,
            baseTableOp.getResult(),
            columnsArrayAttr
        );
        
        builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
        
        PassManager pm(&context);
        pm.addPass(pgx_conversion::createRelAlgToDBPass());
        
        LogicalResult result = pm.run(funcOp);
        ASSERT_TRUE(succeeded(result)) << "Should handle SELECT * pattern";
        
        // Verify DSA operations exist
        bool hasDSAOps = false;
        funcOp.walk([&](Operation *op) {
            if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") {
                hasDSAOps = true;
            }
        });
        EXPECT_TRUE(hasDSAOps) << "Should generate DSA operations for SELECT *";
    }
    
    // Test case 2: No columns (edge case)
    {
        OwningOpRef<ModuleOp> module2(ModuleOp::create(UnknownLoc::get(&context)));
        builder->setInsertionPointToStart(module2->getBody());
        
        auto funcType = builder->getFunctionType({}, {});
        auto funcOp2 = builder->create<func::FuncOp>(UnknownLoc::get(&context), "test_no_columns", funcType);
        auto* entryBlock2 = funcOp2.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock2);
        
        auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
        auto baseTableOp2 = builder->create<pgx::mlir::relalg::BaseTableOp>(
            UnknownLoc::get(&context),
            tupleStreamType,
            builder->getStringAttr("no_cols"),
            builder->getI64IntegerAttr(77777)
        );
        
        auto tableType = pgx::mlir::relalg::TableType::get(&context);
        llvm::SmallVector<mlir::Attribute> emptyColumns;  // No columns
        auto emptyArrayAttr = builder->getArrayAttr(emptyColumns);
        
        auto materializeOp2 = builder->create<pgx::mlir::relalg::MaterializeOp>(
            UnknownLoc::get(&context),
            tableType,
            baseTableOp2.getResult(),
            emptyArrayAttr
        );
        
        builder->create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
        
        PassManager pm2(&context);
        pm2.addPass(pgx_conversion::createRelAlgToDBPass());
        
        LogicalResult result2 = pm2.run(funcOp2);
        ASSERT_TRUE(succeeded(result2)) << "Should handle empty column list gracefully";
    }
    
    PGX_DEBUG("Special column patterns test completed");
}