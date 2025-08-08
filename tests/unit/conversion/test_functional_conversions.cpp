// Phase 3a Functional Conversion Tests: RelAlg → DB Dialect
// Tests the RelAlgToDB conversion pass with specific operations:
// - BaseTableOp → GetExternalOp 
// - RelAlg ReturnOp → func::ReturnOp
// This phase does NOT include DB→DSA conversions (that's Phase 3b)

#include "gtest/gtest.h"
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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

// Test ReturnOp handling (should remain unconverted in Phase 4c-1)
TEST_F(FunctionalConversionsTest, TestReturnOpPassThrough) {
    PGX_DEBUG("Testing ReturnOp remains unconverted in Phase 4c-1");
    
    // Create a simple module and function
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_return_passthrough", funcType);
    
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
    
    // In Phase 4c-1, ReturnOp remains unconverted (marked as LEGAL)
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp)) 
        << "RelAlg ReturnOp should remain unconverted in Phase 4c-1";
    EXPECT_FALSE(containsOperation<func::ReturnOp>(funcOp))
        << "No func::ReturnOp should be created in Phase 4c-1";
    
    // Count func::ReturnOp after conversion (should still be 0)
    int funcReturnCountAfter = countOperations<func::ReturnOp>(funcOp);
    EXPECT_EQ(funcReturnCountAfter, 0) << "Should still have no func::ReturnOp in Phase 4c-1";
    
    PGX_DEBUG("ReturnOp pass-through test completed successfully");
}

// Test GetColumnOp handling (should remain unconverted in Phase 3a)
TEST_F(FunctionalConversionsTest, DISABLED_TestGetColumnOpConversion) {
    PGX_DEBUG("Testing GetColumnOp remains unconverted (disabled in Phase 3a)");
    
    // Create a simple module and function for testing
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create function that returns a nullable int32 type (result of GetColumnOp)
    auto nullableI32Type = ::pgx::db::NullableI32Type::get(&context);
    auto funcType = builder->getFunctionType({}, {nullableI32Type});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_get_column", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp to provide tuple source
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(54321));
    
    // Create GetColumnOp that extracts from the tuple stream
    auto getColumnOp = builder->create<::pgx::mlir::relalg::GetColumnOp>(
        builder->getUnknownLoc(),
        nullableI32Type,  // Result type
        builder->getStringAttr("test_column"),  // Column name
        baseTableOp.getResult());  // Tuple input
    
    // Create return with the column value
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), getColumnOp.getResult());
    
    // Verify GetColumnOp exists before conversion
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::GetColumnOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::db::GetFieldOp>(funcOp));
    
    // Verify the module is well-formed before conversion
    EXPECT_TRUE(module.verify().succeeded()) << "Module should verify before conversion";
    
    // Create and run RelAlgToDB conversion pass on the function
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "RelAlgToDB pass should succeed";
    
    // Verify GetColumnOp remains unconverted (pattern disabled in Phase 3a)
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::GetColumnOp>(funcOp)) << "GetColumnOp should remain unconverted";
    EXPECT_FALSE(containsOperation<::pgx::db::GetFieldOp>(funcOp)) << "No GetFieldOp should be created";
    
    // Verify BaseTableOp was still converted to GetExternalOp
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp)) << "BaseTableOp should be converted";
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp)) << "GetExternalOp should exist";
    
    PGX_DEBUG("GetColumnOp remains unconverted test completed successfully");
}

// Test MaterializeOp handling in Phase 3a - Should remain unconverted per LingoDB architecture
TEST_F(FunctionalConversionsTest, DISABLED_TestMaterializeOpConversion) {
    PGX_DEBUG("Starting MaterializeOp architectural compliance test - should remain unconverted in Phase 3a");
    
    // Create a simple function with MaterializeOp
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // MaterializeOp should remain as TableType since it's not converted in Phase 3a
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_materialize_legal", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a simple BaseTableOp to provide tuple stream for MaterializeOp 
    // This avoids using UnrealizedConversionCastOp which may be causing segfaults
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(99999));
    
    // Create MaterializeOp with the dummy stream - use existing tableType variable
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
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::MaterializeOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    
    // Create and run RelAlgToDB conversion pass on the function
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // The RelAlgToDBPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "RelAlgToDB pass should succeed with MaterializeOp remaining legal";
    
    // CRITICAL: Verify MaterializeOp remains unconverted per LingoDB architecture
    // MaterializeOp belongs in Phase 3b (DB→DSA), not Phase 3a (RelAlg→DB)
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::MaterializeOp>(funcOp)) 
        << "MaterializeOp should remain LEGAL (unconverted) in Phase 3a per LingoDB research";
    
    // Verify RelAlg ReturnOp remains unconverted in Phase 4c-1
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp))
        << "RelAlg ReturnOp should remain unconverted in Phase 4c-1";
    EXPECT_FALSE(containsOperation<func::ReturnOp>(funcOp))
        << "No func::ReturnOp should be created in Phase 4c-1";
    
    // Verify BaseTableOp was converted to GetExternalOp (this conversion still works)
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp))
        << "BaseTableOp should be converted in Phase 3a";
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp))
        << "BaseTableOp should be converted to GetExternalOp";
    
    // No cast operations should be needed since MaterializeOp isn't converted
    EXPECT_FALSE(containsOperation<mlir::UnrealizedConversionCastOp>(funcOp))
        << "No UnrealizedConversionCastOp should be created when MaterializeOp remains legal";
    
    PGX_DEBUG("MaterializeOp architectural compliance test completed successfully - MaterializeOp remains legal in Phase 3a");
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
    
    // Apply Phase 3a conversion on the function (RelAlg → DB only)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // The RelAlgToDBPass operates on func::FuncOp, so run it on the function
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "Phase 3a RelAlgToDB pass should succeed";
    
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
    
    // Create RelAlg ReturnOp (remains unconverted in Phase 4c-1)
    builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), ValueRange{});
    
    // Verify initial state - should have RelAlg operations
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp));
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp));
    EXPECT_FALSE(containsOperation<::pgx::db::GetExternalOp>(funcOp));
    EXPECT_FALSE(containsOperation<func::ReturnOp>(funcOp));
    
    // Apply RelAlgToDB conversion (Phase 4c-1 partial conversion)
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    EXPECT_TRUE(result.succeeded()) << "Phase 4c-1 partial conversion should succeed";
    
    // Verify partial conversion results for Phase 4c-1:
    // - BaseTableOp should be converted to GetExternalOp
    // - ReturnOp should remain unconverted (LEGAL in Phase 4c-1)
    // - Other operations (like MaterializeOp, GetColumnOp) would remain legal if present
    EXPECT_FALSE(containsOperation<::pgx::mlir::relalg::BaseTableOp>(funcOp)) 
        << "BaseTableOp should be converted in Phase 4c-1";
    EXPECT_TRUE(containsOperation<::pgx::mlir::relalg::ReturnOp>(funcOp)) 
        << "ReturnOp should remain unconverted in Phase 4c-1";
    EXPECT_TRUE(containsOperation<::pgx::db::GetExternalOp>(funcOp)) 
        << "GetExternalOp should be created from BaseTableOp";
    EXPECT_FALSE(containsOperation<func::ReturnOp>(funcOp)) 
        << "No func::ReturnOp should be created in Phase 4c-1";
    
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