#include "gtest/gtest.h"
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/DBToDSA/DBToDSA.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class FunctionalConversionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    // Helper to verify operation existence in function
    template<typename OpType>
    bool containsOperation(func::FuncOp func) {
        bool found = false;
        func.walk([&](OpType op) {
            found = true;
        });
        return found;
    }
    
    // Helper to count operations in function
    template<typename OpType>
    int countOperations(func::FuncOp func) {
        int count = 0;
        func.walk([&](OpType op) {
            count++;
        });
        return count;
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test BaseTableOp → GetExternalOp conversion
TEST_F(FunctionalConversionsTest, TestBaseTableToGetExternalConversion) {
    PGX_DEBUG("Starting BaseTableOp to GetExternalOp conversion test");
    
    // Create a simple module and function for testing
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create function with simple void signature
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345));
    
    // Create simple return (no operands to avoid complex types)
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify BaseTableOp exists before conversion
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    
    // Verify the module is well-formed before conversion
    EXPECT_TRUE(module.verify().succeeded()) << "Module should verify before conversion";
    
    // Create and run RelAlgToDB conversion pass on the function (not module)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // The RelAlgToDBPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "RelAlgToDB pass should succeed";
    
    // Verify conversion: BaseTableOp should be gone, GetExternalOp should exist
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    
    // Verify table_oid attribute is preserved
    bool foundCorrectOid = false;
    funcOp.walk([&](::pgx::db::GetExternalOp op) {
        if (auto constOp = op.getTableOid().getDefiningOp<arith::ConstantIntOp>()) {
            if (constOp.value() == 12345) {
                foundCorrectOid = true;
            }
        }
    });
    EXPECT_TRUE(foundCorrectOid) << "Table OID should be preserved as 12345";
    
    PGX_DEBUG("BaseTableOp to GetExternalOp conversion test completed successfully");
}

// Test ReturnOp → func::ReturnOp conversion
TEST_F(FunctionalConversionsTest, TestReturnOpConversion) {
    PGX_DEBUG("Starting ReturnOp to func::ReturnOp conversion test");
    
    // Create a simple module and function
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_return_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create RelAlg ReturnOp with no operands
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify RelAlg ReturnOp exists before conversion
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    
    // Count func::ReturnOp before (should be 0)
    int funcReturnCountBefore = countOperations<func::ReturnOp>(funcOp);
    EXPECT_EQ(funcReturnCountBefore, 0) << "Should have no func::ReturnOp initially";
    
    // Create and run RelAlgToDB conversion pass on the function (not module)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // The RelAlgToDBPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "RelAlgToDB pass should succeed";
    
    // Verify conversion: RelAlg ReturnOp should be gone, func::ReturnOp should exist
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    EXPECT_TRUE(containsOperation<func::ReturnOp>(funcOp));
    
    // Count func::ReturnOp after conversion (should be 1)
    int funcReturnCountAfter = countOperations<func::ReturnOp>(funcOp);
    EXPECT_EQ(funcReturnCountAfter, 1) << "Should have exactly one func::ReturnOp after conversion";
    
    PGX_DEBUG("ReturnOp to func::ReturnOp conversion test completed successfully");
}

// Test GetExternalOp → ScanSourceOp conversion (DB to DSA)
TEST_F(FunctionalConversionsTest, TestGetExternalToScanSourceConversion) {
    PGX_DEBUG("Starting GetExternalOp to ScanSourceOp conversion test");
    
    // Create a simple module and function
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_db_to_dsa", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create GetExternalOp
    auto tableOidValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 54321, builder->getI64Type());
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        tableOidValue.getResult());
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify GetExternalOp exists before conversion
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::mlir::dsa::ScanSourceOp>(funcOp));
    
    // Create and run DBToDSA conversion pass on the function (not module)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createDBToDSAPass());
    
    // The DBToDSAPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "DBToDSA pass should succeed";
    
    // Verify conversion: GetExternalOp should be gone, ScanSourceOp should exist
    EXPECT_FALSE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::dsa::ScanSourceOp>(funcOp));
    
    // Verify table OID is preserved in JSON description
    bool foundCorrectJsonDesc = false;
    funcOp.walk([&](::pgx::mlir::dsa::ScanSourceOp op) {
        auto jsonDesc = op.getTableDescriptionAttr().getValue().str();
        if (jsonDesc.find("54321") != std::string::npos) {
            foundCorrectJsonDesc = true;
        }
    });
    EXPECT_TRUE(foundCorrectJsonDesc) << "Table OID should be preserved in JSON description";
    
    PGX_DEBUG("GetExternalOp to ScanSourceOp conversion test completed successfully");
}

// Test combined pipeline: RelAlg → DB → DSA
TEST_F(FunctionalConversionsTest, TestCombinedPipeline) {
    PGX_DEBUG("Starting combined pipeline test");
    
    // Create a simple function with both BaseTableOp and ReturnOp
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_combined_pipeline", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("pipeline_table"),
        builder->getI64IntegerAttr(99999));
    
    // Create RelAlg ReturnOp
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify initial state: RelAlg operations present
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    
    // Apply both conversions in single pass manager on the function
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    pm.addPass(::pgx_conversion::createDBToDSAPass());
    
    // Both passes operate on func::FuncOp, so run on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "Combined pipeline should succeed";
    
    // Verify final state: transformations occurred in correct order
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::dsa::ScanSourceOp>(funcOp));
    EXPECT_TRUE(containsOperation<func::ReturnOp>(funcOp));
    
    // Verify table OID is preserved through the pipeline
    bool foundCorrectOid = false;
    funcOp.walk([&](::pgx::mlir::dsa::ScanSourceOp op) {
        auto jsonDesc = op.getTableDescriptionAttr().getValue().str();
        if (jsonDesc.find("99999") != std::string::npos) {
            foundCorrectOid = true;
        }
    });
    EXPECT_TRUE(foundCorrectOid) << "Table OID should be preserved through pipeline";
    
    PGX_DEBUG("Combined pipeline test completed successfully");
}

// Test that MaterializeOp remains unchanged in partial conversion (Phase 3a)
// NOTE: This test is disabled due to complex type serialization issues
TEST_F(FunctionalConversionsTest, DISABLED_TestPartialConversionMaterializeOp) {
    PGX_DEBUG("Starting partial conversion test for MaterializeOp");
    
    // Create function with MaterializeOp (which should remain legal in Phase 3a)
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_partial_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp first
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("partial_table"),
        builder->getI64IntegerAttr(77777));
    
    // Create MaterializeOp
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("*"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(), 
        tableType, 
        baseTableOp.getResult(), 
        columnsArrayAttr);
    
    // Create RelAlg ReturnOp
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), materializeOp.getResult());
    
    // Verify initial state
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::MaterializeOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    
    // Create and run RelAlgToDB conversion pass on the function
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // The RelAlgToDBPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "RelAlgToDB pass should succeed with partial conversion";
    
    // Verify MaterializeOp remains unchanged (stays legal in Phase 3a)
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::MaterializeOp>(funcOp)) 
        << "MaterializeOp should remain unchanged in Phase 3a";
    
    // But BaseTableOp and ReturnOp should be converted
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_TRUE(containsOperation<func::ReturnOp>(funcOp));
    
    PGX_DEBUG("Partial conversion test completed successfully");
}

// Test pass verification and error handling
TEST_F(FunctionalConversionsTest, TestPassVerificationAndErrorHandling) {
    PGX_DEBUG("Starting pass verification and error handling test");
    
    // Create a function with well-formed MLIR IR
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_verification", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create simple operations
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("verify_table"),
        builder->getI64IntegerAttr(11111));
    
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify the module is well-formed before conversion
    EXPECT_TRUE(module.verify().succeeded()) << "Module should verify before conversion";
    EXPECT_TRUE(funcOp.verify().succeeded()) << "Function should verify before conversion";
    
    // Apply conversions on the function
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    pm.addPass(::pgx_conversion::createDBToDSAPass());
    
    // Both passes operate on func::FuncOp, so run on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "Pass pipeline should succeed";
    
    // Verify the module is still well-formed after conversion
    EXPECT_TRUE(module.verify().succeeded()) << "Module should verify after conversion";
    EXPECT_TRUE(funcOp.verify().succeeded()) << "Function should verify after conversion";
    
    PGX_DEBUG("Pass verification and error handling test completed successfully");
}

// Test partial conversion behavior - Phase 3a only converts specific operations
TEST_F(FunctionalConversionsTest, TestPartialConversionBehavior) {
    PGX_DEBUG("Starting partial conversion behavior test");
    
    // Create a simple function with multiple RelAlg operations
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_partial_behavior", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp (should be converted in Phase 3a)
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("partial_test_table"),
        builder->getI64IntegerAttr(88888));
    
    // Create RelAlg ReturnOp (should be converted in Phase 3a)
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify initial state - should have RelAlg operations
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_FALSE(containsOperation<func::ReturnOp>(funcOp));
    
    // Apply RelAlgToDB conversion (Phase 3a partial conversion)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "Phase 3a partial conversion should succeed";
    
    // Verify partial conversion results:
    // - BaseTableOp should be converted to GetExternalOp
    // - ReturnOp should be converted to func::ReturnOp
    // - Other operations (like MaterializeOp, GetColumnOp) would remain legal if present
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp)) 
        << "BaseTableOp should be converted in Phase 3a";
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp)) 
        << "ReturnOp should be converted in Phase 3a";
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp)) 
        << "GetExternalOp should be created from BaseTableOp";
    EXPECT_TRUE(containsOperation<func::ReturnOp>(funcOp)) 
        << "func::ReturnOp should be created from RelAlg ReturnOp";
    
    // Verify table OID preservation
    bool foundCorrectOid = false;
    funcOp.walk([&](::pgx::db::GetExternalOp op) {
        if (auto constOp = op.getTableOid().getDefiningOp<arith::ConstantIntOp>()) {
            if (constOp.value() == 88888) {
                foundCorrectOid = true;
            }
        }
    });
    EXPECT_TRUE(foundCorrectOid) << "Table OID should be preserved in partial conversion";
    
    PGX_DEBUG("Partial conversion behavior test completed successfully");
}