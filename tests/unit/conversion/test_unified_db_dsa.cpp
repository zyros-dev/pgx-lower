#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "gtest/gtest.h"

using namespace mlir;

class UnifiedDBDSATest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<mlir::db::DBDialect>();
        context.getOrLoadDialect<mlir::dsa::DSADialect>();
        context.getOrLoadDialect<mlir::util::UtilDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
        
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
        builder = std::make_unique<mlir::OpBuilder>(module.getBodyRegion());
    }

    void TearDown() override {
        module.erase();
    }

    MLIRContext context;
    ModuleOp module;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(UnifiedDBDSATest, ProcessesMixedDBAndDSAOperations) {
    // Create a function with mixed DB and DSA operations
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(builder->getUnknownLoc(), "test_mixed_ops", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create DB operations
    auto i32Type = builder->getI32Type();
    auto dbConstant = builder->create<mlir::db::ConstantOp>(
        builder->getUnknownLoc(), 
        i32Type, 
        builder->getI32IntegerAttr(42)
    );
    
    // Create DSA operation with DB type
    auto dsaType = mlir::dsa::RecordType::get(&context, {});
    auto createOp = builder->create<mlir::dsa::CreateOp>(
        builder->getUnknownLoc(),
        dsaType
    );
    
    // Create a DB runtime call that uses DSA types
    auto stringType = mlir::db::StringType::get(&context);
    auto runtimeCall = builder->create<mlir::db::RuntimeCall>(
        builder->getUnknownLoc(),
        TypeRange{stringType},
        "test_function",
        ValueRange{createOp}
    );
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Run the unified DBToStd pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createDBToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    
    // Verify the module is still valid
    ASSERT_TRUE(succeeded(verify(module)));
    
    // Check that both DB and DSA operations were converted
    bool hasArithOps = false;
    bool hasMemRefOps = false;
    
    module.walk([&](Operation* op) {
        if (isa<arith::ArithDialect>(op->getDialect())) {
            hasArithOps = true;
        }
        if (op->getDialect() && op->getDialect()->getNamespace() == "memref") {
            hasMemRefOps = true;
        }
    });
    
    EXPECT_TRUE(hasArithOps) << "DB operations should be converted to Arith dialect";
    EXPECT_TRUE(hasMemRefOps) << "DSA operations should be converted to MemRef dialect";
}

TEST_F(UnifiedDBDSATest, HandlesComplexTypeConversions) {
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(builder->getUnknownLoc(), "test_type_conversion", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a complex scenario with nested types
    auto stringType = mlir::db::StringType::get(&context);
    auto recordType = mlir::dsa::RecordType::get(&context, {stringType});
    
    // DB operation producing DSA-compatible type
    auto dbNullOp = builder->create<mlir::db::NullOp>(
        builder->getUnknownLoc(),
        recordType
    );
    
    // DSA operation consuming DB-produced value
    auto dsaAt = builder->create<mlir::dsa::At>(
        builder->getUnknownLoc(),
        stringType,
        dbNullOp,
        builder->create<arith::ConstantIndexOp>(builder->getUnknownLoc(), 0)
    );
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Run the unified pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createDBToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    ASSERT_TRUE(succeeded(verify(module)));
    
    // Verify no DB or DSA operations remain
    bool hasDBOps = false;
    bool hasDSAOps = false;
    
    module.walk([&](Operation* op) {
        if (isa<mlir::db::DBDialect>(op->getDialect())) {
            hasDBOps = true;
        }
        if (isa<mlir::dsa::DSADialect>(op->getDialect())) {
            hasDSAOps = true;
        }
    });
    
    EXPECT_FALSE(hasDBOps) << "All DB operations should be lowered";
    EXPECT_FALSE(hasDSAOps) << "All DSA operations should be lowered";
}

TEST_F(UnifiedDBDSATest, PreservesControlFlowWithMixedOperations) {
    auto funcType = builder->getFunctionType({builder->getI1Type()}, {});
    auto func = builder->create<func::FuncOp>(builder->getUnknownLoc(), "test_control_flow", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    auto condition = entryBlock->getArgument(0);
    
    // Create SCF if with mixed operations
    auto ifOp = builder->create<scf::IfOp>(
        builder->getUnknownLoc(),
        TypeRange{},
        condition,
        [&](OpBuilder& b, Location loc) {
            // True branch with DB operation
            auto dbConst = b.create<mlir::db::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(1));
            b.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder& b, Location loc) {
            // False branch with DSA operation
            auto dsaRecord = b.create<mlir::dsa::CreateOp>(
                loc, 
                mlir::dsa::RecordType::get(&context, {})
            );
            b.create<scf::YieldOp>(loc);
        }
    );
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Run the unified pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createDBToStdPass());
    
    ASSERT_TRUE(succeeded(pm.run(module)));
    ASSERT_TRUE(succeeded(verify(module)));
    
    // Verify control flow structure is preserved
    bool hasControlFlow = false;
    module.walk([&](scf::IfOp) {
        hasControlFlow = true;
    });
    
    EXPECT_TRUE(hasControlFlow) << "Control flow structure should be preserved";
}

TEST_F(UnifiedDBDSATest, HandlesRuntimeCallsWithDSATypes) {
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(builder->getUnknownLoc(), "test_runtime_calls", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create DSA record to pass to runtime call
    auto recordType = mlir::dsa::RecordType::get(&context, {builder->getI32Type()});
    auto record = builder->create<mlir::dsa::CreateOp>(builder->getUnknownLoc(), recordType);
    
    // DB runtime call accepting DSA type
    auto stringType = mlir::db::StringType::get(&context);
    auto runtimeCall = builder->create<mlir::db::RuntimeCall>(
        builder->getUnknownLoc(),
        TypeRange{stringType},
        "record_to_string",
        ValueRange{record}
    );
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Run the unified pass
    PassManager pm(&context);
    pm.addPass(mlir::db::createDBToStdPass());
    
    // Should succeed even with complex type interactions
    ASSERT_TRUE(succeeded(pm.run(module)));
    ASSERT_TRUE(succeeded(verify(module)));
}