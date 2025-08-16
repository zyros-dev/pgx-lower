#include <gtest/gtest.h>
#include "frontend/SQL/postgresql_ast_translator.h"
#include "execution/logging.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "runtime/metadata.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// Mock PostgreSQL structures for testing
extern "C" {
struct Plan {
    int type;
    int startup_cost;
    int total_cost;
    int plan_rows;
    int plan_width;
};

struct Scan {
    Plan plan;
    unsigned int scanrelid;
};

struct SeqScan {
    Scan scan;
};

struct PlannedStmt {
    int type;
    int commandType;
    Plan* planTree;
    // Add other fields as needed for tests
};
}

class BaseTableOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        
        // Register required dialects
        context->getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        context->getOrLoadDialect<mlir::dsa::DSADialect>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        
        builder = std::make_unique<mlir::OpBuilder>(context.get());
        
        // Create a module to hold our operations
        module = mlir::ModuleOp::create(builder->getUnknownLoc());
        builder->setInsertionPointToStart(module.getBody());
    }
    
    void TearDown() override {
        // Cleanup happens automatically through unique_ptr
    }
    
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<mlir::OpBuilder> builder;
    mlir::ModuleOp module;
};

TEST_F(BaseTableOpTest, CreateBaseTableOpWithMetadata) {
    PGX_INFO("Testing BaseTableOp creation with metadata");
    
    // Create table metadata
    auto tableMetaData = std::make_shared<runtime::TableMetaData>();
    tableMetaData->setNumRows(100);
    
    // Create TableMetaDataAttr
    auto tableMetaAttr = mlir::relalg::TableMetaDataAttr::get(
        context.get(),
        tableMetaData
    );
    
    // Get column manager
    auto& columnManager = context->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    
    // Create column definitions
    auto idColumn = columnManager.createDef("test_table", "id");
    if (idColumn.getColumnPtr()) {
        idColumn.getColumn().type = builder->getI32Type();
    }
    
    auto nameColumn = columnManager.createDef("test_table", "name");
    if (nameColumn.getColumnPtr()) {
        nameColumn.getColumn().type = builder->getI64Type();  // Using i64 to represent string for now
    }
    
    // Create dictionary attribute with columns
    std::vector<mlir::NamedAttribute> namedAttrs;
    namedAttrs.push_back(builder->getNamedAttr("id", idColumn));
    namedAttrs.push_back(builder->getNamedAttr("name", nameColumn));
    auto columnsAttr = builder->getDictionaryAttr(namedAttrs);
    
    // Create BaseTableOp
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        mlir::relalg::TupleStreamType::get(context.get()),
        builder->getStringAttr("test_table|oid:16384"),
        tableMetaAttr,
        columnsAttr
    );
    
    ASSERT_NE(baseTableOp, nullptr);
    
    // Verify the operation can be printed without crashing
    std::string mlirStr;
    llvm::raw_string_ostream stream(mlirStr);
    baseTableOp->print(stream);
    mlirStr = stream.str();
    
    PGX_INFO("Generated MLIR: " + mlirStr);
    
    // Verify expected content in the printed MLIR
    EXPECT_TRUE(mlirStr.find("relalg.basetable") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("test_table|oid:16384") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("columns:") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("id") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("name") != std::string::npos);
    
    PGX_INFO("BaseTableOp created and printed successfully without crash");
}

TEST_F(BaseTableOpTest, CreateBaseTableOpWithNullMetadata) {
    PGX_INFO("Testing BaseTableOp creation with null metadata");
    
    // Create TableMetaDataAttr with null pointer (edge case)
    auto tableMetaAttr = mlir::relalg::TableMetaDataAttr::get(
        context.get(),
        nullptr  // Null metadata to test our fix
    );
    
    // Get column manager
    auto& columnManager = context->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    
    // Create minimal column definition
    auto idColumn = columnManager.createDef("empty_table", "id");
    if (idColumn.getColumnPtr()) {
        idColumn.getColumn().type = builder->getI32Type();
    }
    
    // Create dictionary attribute with column
    std::vector<mlir::NamedAttribute> namedAttrs;
    namedAttrs.push_back(builder->getNamedAttr("id", idColumn));
    auto columnsAttr = builder->getDictionaryAttr(namedAttrs);
    
    // Create BaseTableOp with null metadata
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        mlir::relalg::TupleStreamType::get(context.get()),
        builder->getStringAttr("empty_table|oid:0"),
        tableMetaAttr,
        columnsAttr
    );
    
    ASSERT_NE(baseTableOp, nullptr);
    
    // Verify the operation can be printed without crashing (this tests our fix)
    std::string mlirStr;
    llvm::raw_string_ostream stream(mlirStr);
    baseTableOp->print(stream);
    mlirStr = stream.str();
    
    PGX_INFO("Generated MLIR with null metadata: " + mlirStr);
    
    // Verify it still contains the basic structure
    EXPECT_TRUE(mlirStr.find("relalg.basetable") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("empty_table|oid:0") != std::string::npos);
    
    PGX_INFO("BaseTableOp with null metadata handled gracefully");
}

TEST_F(BaseTableOpTest, SeqScanTranslationWithProperBaseTableOp) {
    PGX_INFO("Testing SeqScan translation with proper BaseTableOp");
    
    // Create a mock SeqScan node
    SeqScan seqScan;
    seqScan.scan.plan.type = 17;  // T_SeqScan in test environment
    seqScan.scan.plan.startup_cost = 0;
    seqScan.scan.plan.total_cost = 100;
    seqScan.scan.plan.plan_rows = 1000;
    seqScan.scan.plan.plan_width = 32;
    seqScan.scan.scanrelid = 1;  // References table with scanrelid 1
    
    // Create translator
    auto translator = postgresql_ast::createPostgreSQLASTTranslator(*context);
    
    // Create a simple PlannedStmt for testing - use the global struct
    ::PlannedStmt stmt;
    stmt.type = 0;
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = &seqScan.scan.plan;
    
    // Translate the query
    auto modulePtr = translator->translateQuery(&stmt);
    ASSERT_NE(modulePtr, nullptr);
    
    // Verify the module contains expected operations
    std::string mlirStr;
    llvm::raw_string_ostream stream(mlirStr);
    (*modulePtr)->print(stream);
    mlirStr = stream.str();
    
    PGX_INFO("Translated MLIR:\n" + mlirStr);
    
    // Verify BaseTableOp was created properly
    EXPECT_TRUE(mlirStr.find("relalg.basetable") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("table_identifier = \"test|oid:16384\"") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("columns:") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("@test::@id") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("relalg.materialize") != std::string::npos);
    
    PGX_INFO("SeqScan successfully translated to BaseTableOp");
}