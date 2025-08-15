#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"
#include "runtime/metadata.h"
#include "runtime/tuple_access.h"

// PostgreSQL OID type for test
typedef unsigned int Oid;

namespace {

class SequentialPipelinesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create context with all dialects
        context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                       mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                       mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                       mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                       mlir::dsa::DSADialect, mlir::util::UtilDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
    }
    
    std::shared_ptr<mlir::MLIRContext> context;
};

// Test that the StandardToLLVM pass can be created without DataLayoutAnalysis issues
TEST_F(SequentialPipelinesTest, TestStandardToLLVMPassCreation) {
    // Create a simple module with standard operations
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create a simple function with standard operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add a simple arithmetic operation
    auto c1 = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 32);
    auto c2 = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 2, 32);
    auto add = builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), c1, c2);
    
    // Add return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Run StandardToLLVM pipeline
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    
    // This should not crash with DataLayoutAnalysis issues
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
}

// Test the complete sequential pipeline approach
TEST_F(SequentialPipelinesTest, TestSequentialPipelineExecution) {
    // Create a module that simulates what would come from RelAlg→DB
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    
    // Initialize UtilDialect function helper
    auto* utilDialect = context->getOrLoadDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    utilDialect->getFunctionHelper().setParentModule(module);
    
    // Create a function that represents DB operations
    mlir::OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "query_main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add return (simulating empty query for now)
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Test Phase 2: DB+DSA→Standard
    {
        PGX_INFO("Testing Phase 2: DB+DSA→Standard pipeline");
        mlir::PassManager pm2(context.get());
        mlir::pgx_lower::createDBDSAToStandardPipeline(pm2, true);
        auto result = pm2.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
    }
    
    // Test Phase 3: Standard→LLVM
    {
        PGX_INFO("Testing Phase 3: Standard→LLVM pipeline");
        mlir::PassManager pm3(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm3, true);
        auto result = pm3.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
    }
    
    // Verify final module is valid LLVM
    EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
}

// Test that replicates PostgreSQL's exact MLIR operations from Test 1
TEST_F(SequentialPipelinesTest, TestPostgreSQLLikeOperations) {
    PGX_INFO("Testing PostgreSQL-like operations with real table metadata");
    
    // Create module
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    
    // Initialize UtilDialect function helper
    auto* utilDialect = context->getOrLoadDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    utilDialect->getFunctionHelper().setParentModule(module);
    
    // Create main function
    mlir::OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "query_main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Replicate PostgreSQL's BaseTableOp creation
    // From Test 1 output: Table OID 5135218, column "id", type OID 23 (INT4)
    std::string tableName = "test";
    Oid tableOid = 5135218;  // Actual OID from PostgreSQL output
    std::string tableIdentifier = tableName + "|oid:" + std::to_string(tableOid);
    
    PGX_INFO("Creating BaseTableOp with PostgreSQL metadata: " + tableIdentifier);
    
    // Create table metadata matching PostgreSQL's structure
    auto tableMetaData = std::make_shared<runtime::TableMetaData>();
    
    // Add column metadata for "id" column (type OID 23 = INT4)
    auto idColumnMetaData = std::make_shared<runtime::ColumnMetaData>();
    runtime::ColumnType int4Type;
    int4Type.base = "int4";  // PostgreSQL INT4 type
    int4Type.nullable = false;
    idColumnMetaData->setColumnType(int4Type);
    tableMetaData->addColumn("id", idColumnMetaData);
    tableMetaData->setNumRows(1);  // Simulating one row
    
    auto tableMetaDataAttr = mlir::relalg::TableMetaDataAttr::get(context.get(), tableMetaData);
    
    // Create column definitions
    auto& columnManager = context->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    
    // Create column definition for "id" with INT4 type
    auto idColumnDef = columnManager.createDef("test", "id");
    // CRITICAL: Set the type on the column to avoid null pointer crash
    idColumnDef.getColumn().type = builder.getI32Type();  // INT4 maps to i32
    
    // Create columns dictionary attribute
    mlir::NamedAttribute columnEntry(
        builder.getStringAttr("id"),
        idColumnDef
    );
    auto columnsAttr = builder.getDictionaryAttr({columnEntry});
    
    // Create BaseTableOp
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(context.get());
    auto baseTableOp = builder.create<mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr(tableIdentifier),
        tableMetaDataAttr,
        columnsAttr
    );
    
    PGX_INFO("BaseTableOp created, now creating MaterializeOp");
    
    // Create MaterializeOp to match PostgreSQL's pattern
    // Create column references for materialization
    auto idColumnRef = columnManager.createRef(&idColumnDef.getColumn());
    std::vector<mlir::Attribute> columnRefAttrs = {idColumnRef};
    std::vector<mlir::Attribute> columnNameAttrs = {builder.getStringAttr("id")};
    
    auto columnRefs = builder.getArrayAttr(columnRefAttrs);
    auto columnNames = builder.getArrayAttr(columnNameAttrs);
    
    // Create MaterializeOp
    auto tableType = mlir::dsa::TableType::get(context.get());
    auto materializeOp = builder.create<mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnRefs,
        columnNames
    );
    
    PGX_INFO("MaterializeOp created, adding return");
    
    // For now, return constant like PostgreSQL does
    auto constantOp = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constantOp.getResult());
    
    // Print module state before pipeline
    PGX_INFO("Module before pipeline (skipping print due to TupleStreamType print issue)");
    // TODO: Fix TupleStreamType printing
    // module->print(llvm::errs());
    // llvm::errs() << "\n";
    
    // Count operations before pipeline (should be 6 like PostgreSQL)
    size_t opCount = 0;
    module->walk([&](mlir::Operation* op) {
        opCount++;
        PGX_INFO("Operation: " + std::string(op->getName().getStringRef()));
    });
    PGX_INFO("Total operations before pipeline: " + std::to_string(opCount));
    EXPECT_EQ(opCount, 6) << "Should have 6 operations like PostgreSQL";
    
    // Follow PostgreSQL's exact pipeline order
    // Phase 3a: RelAlg→DB conversion MUST happen first
    {
        PGX_INFO("Running Phase 3a: RelAlg→DB pipeline");
        mlir::PassManager relalgToDbPM(context.get());
        mlir::pgx_lower::createRelAlgToDBPipeline(relalgToDbPM, true);
        
        auto phase3aResult = relalgToDbPM.run(module);
        if (mlir::failed(phase3aResult)) {
            PGX_ERROR("Phase 3a (RelAlg→DB) failed!");
            FAIL() << "RelAlg→DB conversion failed - crash is in Phase 3a";
        }
        
        PGX_INFO("Phase 3a succeeded - RelAlg operations converted to DB operations");
        
        // Count operations after Phase 3a
        size_t dbOpCount = 0;
        module->walk([&](mlir::Operation* op) {
            dbOpCount++;
            PGX_INFO("After Phase 3a: " + std::string(op->getName().getStringRef()));
        });
        PGX_INFO("Operations after Phase 3a: " + std::to_string(dbOpCount));
    }
    
    // Phase 3b: DB+DSA→Standard conversion on the DB operations
    {
        PGX_INFO("Running Phase 3b: DB+DSA→Standard pipeline");
        mlir::PassManager dbToStdPM(context.get());
        mlir::pgx_lower::createDBDSAToStandardPipeline(dbToStdPM, true);
        
        auto phase3bResult = dbToStdPM.run(module);
        if (mlir::succeeded(phase3bResult)) {
            PGX_INFO("Phase 3b succeeded! Complete pipeline works in unit test");
            
            // Verify module is still valid
            EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
            
            // Count final operations
            size_t finalOpCount = 0;
            module->walk([&](mlir::Operation* op) {
                finalOpCount++;
                PGX_INFO("After Phase 3b: " + std::string(op->getName().getStringRef()));
            });
            PGX_INFO("Final operations after Phase 3b: " + std::to_string(finalOpCount));
        } else {
            PGX_ERROR("Phase 3b (DB→Std) failed - crash is in Phase 3b with DB operations!");
            FAIL() << "DB→Std conversion failed on DB operations";
        }
    }
}

} // namespace