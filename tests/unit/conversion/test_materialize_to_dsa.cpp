#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/DBToDSA/DBToDSA.h"

#include "execution/logging.h"
#include "test_helpers.h"

using namespace mlir;

namespace {

class MaterializeToDSATest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("Setting up MaterializeToDSATest");
        
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }

    // Helper function to collect operations of a specific type
    template<typename OpType>
    std::vector<OpType> collectOpsOfType(Operation* op) {
        std::vector<OpType> ops;
        op->walk([&](OpType specificOp) {
            ops.push_back(specificOp);
        });
        return ops;
    }
    
    // Helper function to count all operations in a module
    size_t countAllOpsInModule(ModuleOp module) {
        size_t count = 0;
        module.walk([&](Operation* op) {
            count++;
        });
        return count;
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test that MaterializeOp converts to DSA CreateDS→Finalize sequence
TEST_F(MaterializeToDSATest, MaterializeOpConvertsToCreateDSFinalize) {
    PGX_DEBUG("Testing MaterializeOp conversion to DSA CreateDS→Finalize sequence");
    
    // Create module and function
    auto moduleOp = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_materialize", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create RelAlg TableType for MaterializeOp
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    
    // Create a BaseTableOp as source for MaterializeOp  
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345)
    );
    
    // BaseTableOp no longer has a body - it's a simple operation
    // Create MaterializeOp with column specification
    builder->setInsertionPointAfter(baseTableOp);
    auto columnNames = builder->getArrayAttr({builder->getStringAttr("id")});
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnNames
    );
    
    // Add return operation
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    PGX_DEBUG("Created MaterializeOp, verifying operation before conversion");
    
    // Verify MaterializeOp exists before conversion
    auto materializes = collectOpsOfType<::pgx::mlir::relalg::MaterializeOp>(funcOp);
    EXPECT_EQ(materializes.size(), 1) << "MaterializeOp should exist before conversion";
    
    // Verify no DSA operations exist before conversion
    auto createDSBefore = collectOpsOfType<::pgx::mlir::dsa::CreateDSOp>(funcOp);
    auto finalizeBefore = collectOpsOfType<::pgx::mlir::dsa::FinalizeOp>(funcOp);
    EXPECT_EQ(createDSBefore.size(), 0) << "No DSA CreateDS should exist before conversion";
    EXPECT_EQ(finalizeBefore.size(), 0) << "No DSA Finalize should exist before conversion";
    
    PGX_DEBUG("Running DBToDSA conversion pass");
    
    // Dump the module before conversion to check its validity
    std::cout << "Module before conversion:" << std::endl;
    moduleOp.dump();
    std::cout << "Module verification before conversion: " 
              << (moduleOp.verify().succeeded() ? "SUCCESS" : "FAILURE") << std::endl;
    
    // Apply DBToDSA conversion pass
    PassManager passManager(&context);
    // DBToDSA is a function pass, so we need to nest it properly
    passManager.addNestedPass<func::FuncOp>(::mlir::pgx_conversion::createDBToDSAPass());
    
    std::cout << "About to run pass manager..." << std::endl;
    auto result = passManager.run(moduleOp);
    std::cout << "Pass manager completed with result: " 
              << (result.succeeded() ? "SUCCESS" : "FAILURE") << std::endl;
    
    EXPECT_TRUE(result.succeeded()) << "DBToDSA pass should succeed";
    
    PGX_DEBUG("DBToDSA conversion completed, verifying results");
    
    // Verify MaterializeOp is converted (removed)
    auto materializesAfter = collectOpsOfType<::pgx::mlir::relalg::MaterializeOp>(funcOp);
    EXPECT_EQ(materializesAfter.size(), 0) << "MaterializeOp should be converted and removed";
    
    // Verify DSA operations are created
    auto createDSAfter = collectOpsOfType<::pgx::mlir::dsa::CreateDSOp>(funcOp);
    auto finalizeAfter = collectOpsOfType<::pgx::mlir::dsa::FinalizeOp>(funcOp);
    
    EXPECT_EQ(createDSAfter.size(), 1) << "One DSA CreateDS should be created";
    EXPECT_EQ(finalizeAfter.size(), 1) << "One DSA Finalize should be created";
    
    if (!createDSAfter.empty() && !finalizeAfter.empty()) {
        // Verify the sequence: CreateDS → Finalize
        auto createDS = createDSAfter[0];
        auto finalize = finalizeAfter[0];
        
        // Verify CreateDS produces TableBuilder type
        auto createDSResult = createDS.getResult();
        EXPECT_TRUE(createDSResult.getType().isa<::pgx::mlir::dsa::TableBuilderType>())
            << "CreateDS should produce TableBuilderType";
            
        // Verify Finalize consumes TableBuilder and produces Table
        auto finalizeInput = finalize.getBuilder();
        auto finalizeResult = finalize.getResult();
        
        EXPECT_EQ(finalizeInput, createDSResult) 
            << "Finalize should consume CreateDS result";
        EXPECT_TRUE(finalizeResult.getType().isa<::pgx::mlir::dsa::TableType>())
            << "Finalize should produce TableType";
    }
    
    PGX_DEBUG("MaterializeOp conversion test completed successfully");
}

// Test that the conversion preserves module structure
TEST_F(MaterializeToDSATest, ConversionPreservesModuleStructure) {
    PGX_DEBUG("Testing that MaterializeOp conversion preserves module structure");
    
    // Create module with MaterializeOp
    auto moduleOp = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_preservation", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("preservation_table"),
        builder->getI64IntegerAttr(54321)
    );
    
    // BaseTableOp no longer has a body - it's a simple operation
    // Create MaterializeOp
    builder->setInsertionPointAfter(baseTableOp);
    auto columnNames = builder->getArrayAttr({builder->getStringAttr("value")});
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnNames
    );
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify module is valid before conversion
    EXPECT_TRUE(moduleOp.verify().succeeded()) << "Module should be valid before conversion";
    
    // Count operations before conversion
    size_t totalOpsBefore = countAllOpsInModule(moduleOp);
    
    PGX_DEBUG("Module has " + std::to_string(totalOpsBefore) + " operations before conversion");
    
    // Apply conversion
    PassManager passManager(&context);
    // DBToDSA is a function pass, so we need to nest it properly
    passManager.addNestedPass<func::FuncOp>(::mlir::pgx_conversion::createDBToDSAPass());
    
    auto result = passManager.run(moduleOp);
    EXPECT_TRUE(result.succeeded()) << "Conversion should succeed";
    
    // Verify module is still valid after conversion
    EXPECT_TRUE(moduleOp.verify().succeeded()) << "Module should be valid after conversion";
    
    // Count operations after conversion - should have replaced 1 MaterializeOp with 2 DSA ops (net +1)
    size_t totalOpsAfter = countAllOpsInModule(moduleOp);
    
    PGX_DEBUG("Module has " + std::to_string(totalOpsAfter) + " operations after conversion");
    
    // We expect MaterializeOp (1) to be replaced by CreateDS + Finalize (2), so +1 total
    EXPECT_EQ(totalOpsAfter, totalOpsBefore + 1) << "Should have net +1 operations (MaterializeOp → CreateDS + Finalize)";
    
    PGX_DEBUG("Module structure preservation test completed successfully");
}

// Test that GetExternalOp converts to DSA ScanSourceOp
TEST_F(MaterializeToDSATest, GetExternalOpConvertsToScanSource) {
    PGX_DEBUG("Testing GetExternalOp conversion to DSA ScanSourceOp");
    
    // Create module and function
    auto moduleOp = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_get_external", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create GetExternalOp
    auto externalSourceType = ::pgx::db::ExternalSourceType::get(&context);
    auto tableOid = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(),
        builder->getI64IntegerAttr(12345)
    );
    
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        externalSourceType,
        tableOid.getResult()
    );
    
    // Add return operation
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    PGX_DEBUG("Created GetExternalOp, verifying operation before conversion");
    
    // Verify GetExternalOp exists before conversion
    auto getExternals = collectOpsOfType<::pgx::db::GetExternalOp>(funcOp);
    EXPECT_EQ(getExternals.size(), 1) << "GetExternalOp should exist before conversion";
    
    // Verify no DSA ScanSourceOp exists before conversion
    auto scanSourcesBefore = collectOpsOfType<::pgx::mlir::dsa::ScanSourceOp>(funcOp);
    EXPECT_EQ(scanSourcesBefore.size(), 0) << "No DSA ScanSourceOp should exist before conversion";
    
    PGX_DEBUG("Running DBToDSA conversion pass");
    
    // Apply DBToDSA conversion pass
    PassManager passManager(&context);
    passManager.addNestedPass<func::FuncOp>(::mlir::pgx_conversion::createDBToDSAPass());
    
    auto result = passManager.run(moduleOp);
    EXPECT_TRUE(result.succeeded()) << "DBToDSA pass should succeed";
    
    PGX_DEBUG("DBToDSA conversion completed, verifying results");
    
    // Verify GetExternalOp is converted (removed)
    auto getExternalsAfter = collectOpsOfType<::pgx::db::GetExternalOp>(funcOp);
    EXPECT_EQ(getExternalsAfter.size(), 0) << "GetExternalOp should be converted and removed";
    
    // Verify DSA ScanSourceOp is created
    auto scanSourcesAfter = collectOpsOfType<::pgx::mlir::dsa::ScanSourceOp>(funcOp);
    EXPECT_EQ(scanSourcesAfter.size(), 1) << "One DSA ScanSourceOp should be created";
    
    if (!scanSourcesAfter.empty()) {
        auto scanSource = scanSourcesAfter[0];
        
        // Verify ScanSourceOp produces GenericIterableType
        auto scanResult = scanSource.getResult();
        EXPECT_TRUE(scanResult.getType().isa<::pgx::mlir::dsa::GenericIterableType>())
            << "ScanSourceOp should produce GenericIterableType";
            
        // Verify the table description attribute
        auto tableDesc = scanSource.getTableDescription();
        EXPECT_TRUE(tableDesc.contains("postgresql_table"))
            << "ScanSourceOp should have postgresql_table description";
    }
    
    PGX_DEBUG("GetExternalOp conversion test completed successfully");
}

// Test that StreamResultsOp is properly handled (removed as no-op)
TEST_F(MaterializeToDSATest, StreamResultsOpIsRemoved) {
    PGX_DEBUG("Testing StreamResultsOp removal during DBToDSA conversion");
    
    // Create module and function
    auto moduleOp = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_stream_results", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create StreamResultsOp
    auto streamResultsOp = builder->create<::pgx::db::StreamResultsOp>(
        builder->getUnknownLoc()
    );
    
    // Add return operation
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    PGX_DEBUG("Created StreamResultsOp, verifying operation before conversion");
    
    // Verify StreamResultsOp exists before conversion
    auto streamResults = collectOpsOfType<::pgx::db::StreamResultsOp>(funcOp);
    EXPECT_EQ(streamResults.size(), 1) << "StreamResultsOp should exist before conversion";
    
    PGX_DEBUG("Running DBToDSA conversion pass");
    
    // Apply DBToDSA conversion pass
    PassManager passManager(&context);
    passManager.addNestedPass<func::FuncOp>(::mlir::pgx_conversion::createDBToDSAPass());
    
    auto result = passManager.run(moduleOp);
    EXPECT_TRUE(result.succeeded()) << "DBToDSA pass should succeed";
    
    PGX_DEBUG("DBToDSA conversion completed, verifying results");
    
    // Verify StreamResultsOp is removed
    auto streamResultsAfter = collectOpsOfType<::pgx::db::StreamResultsOp>(funcOp);
    EXPECT_EQ(streamResultsAfter.size(), 0) << "StreamResultsOp should be removed";
    
    PGX_DEBUG("StreamResultsOp removal test completed successfully");
}

} // namespace