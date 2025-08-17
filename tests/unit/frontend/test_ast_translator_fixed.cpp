#include <gtest/gtest.h>
#include <cstring>
#include "pgx_lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx_lower/execution/logging.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// PostgreSQL 17 compatible structures
extern "C" {
    typedef int16_t int16;
    typedef int16 AttrNumber;
    typedef unsigned int Oid;
    
    struct ListCell {
        union {
            void* ptr_value;
            int int_value;
            unsigned int oid_value;
        } data;
    };
    
    struct List {
        int type;
        int length;
        int max_length;
        ListCell* elements;
        ListCell* initial_elements;
    };
    
    struct Node {
        int type;
    };
    
    struct Plan {
        int type;
        double startup_cost;
        double total_cost;
        double plan_rows;
        int plan_width;
        bool parallel_aware;
        bool parallel_safe;
        bool async_capable;
        int plan_node_id;
        List* targetlist;
        List* qual;
        Plan* lefttree;
        Plan* righttree;
        List* initPlan;
        void* extParam;
        void* allParam;
    };
    
    struct SeqScan {
        Plan plan;
        struct {
            int scanrelid;
        } scan;
    };
    
    struct PlannedStmt {
        int type;
        int commandType;
        uint64_t queryId;
        bool hasReturning;
        bool hasModifyingCTE;
        bool canSetTag;
        bool transientPlan;
        bool dependsOnRole;
        bool parallelModeNeeded;
        int jitFlags;
        Plan* planTree;
        List* rtable;
    };
    
    #define T_PlannedStmt 67
    #define T_SeqScan 335
    #define lfirst(lc) ((lc)->data.ptr_value)
}

class ASTTranslatorFixedTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        context->loadDialect<mlir::relalg::RelAlgDialect>();
        context->loadDialect<mlir::dsa::DSADialect>();
        context->loadDialect<mlir::util::UtilDialect>();
        context->loadDialect<mlir::db::DBDialect>();
        
        translator = postgresql_ast::createPostgreSQLASTTranslator(*context);
    }
    
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<postgresql_ast::PostgreSQLASTTranslator> translator;
};

TEST_F(ASTTranslatorFixedTest, TranslatesSimpleSeqScan) {
    PGX_INFO("Testing simple SeqScan translation with fixed structures");
    
    // Create a simple SeqScan node
    SeqScan seqScan{};
    memset(&seqScan, 0, sizeof(SeqScan));
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.scan.scanrelid = 1;
    
    // Create PlannedStmt
    PlannedStmt stmt{};
    memset(&stmt, 0, sizeof(PlannedStmt));
    stmt.type = T_PlannedStmt;
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = &seqScan.plan;
    
    // Translate
    auto module = translator->translateQuery(&stmt);
    
    // Verify success
    ASSERT_NE(module, nullptr) << "Translation should succeed with fixed structures";
    
    // Convert to string for validation
    std::string mlirStr;
    llvm::raw_string_ostream stream(mlirStr);
    module->print(stream);
    mlirStr = stream.str();
    
    PGX_INFO("Generated MLIR:\n" + mlirStr);
    
    // Verify expected content
    EXPECT_TRUE(mlirStr.find("func.func @main") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("relalg.basetable") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("relalg.materialize") != std::string::npos);
    EXPECT_TRUE(mlirStr.find("return") != std::string::npos);
    
    PGX_INFO("Simple SeqScan translated successfully with fixed structures");
}

TEST_F(ASTTranslatorFixedTest, HandlesEmptyQualList) {
    PGX_INFO("Testing SeqScan with empty qual list");
    
    // Create a SeqScan with an empty qual list
    SeqScan seqScan{};
    memset(&seqScan, 0, sizeof(SeqScan));
    seqScan.plan.type = T_SeqScan;
    seqScan.plan.startup_cost = 0.0;
    seqScan.plan.total_cost = 10.0;
    seqScan.plan.plan_rows = 100;
    seqScan.plan.plan_width = 32;
    seqScan.scan.scanrelid = 1;
    
    // Create an empty qual list (PostgreSQL 17 style)
    List qualList{};
    qualList.type = 16;  // T_List
    qualList.length = 0;
    qualList.max_length = 0;
    qualList.elements = nullptr;
    seqScan.plan.qual = &qualList;
    
    // Create PlannedStmt
    PlannedStmt stmt{};
    memset(&stmt, 0, sizeof(PlannedStmt));
    stmt.type = T_PlannedStmt;
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = &seqScan.plan;
    
    // Translate - should handle empty qual list gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Verify success
    ASSERT_NE(module, nullptr) << "Translation should handle empty qual list";
    
    PGX_INFO("Empty qual list handled successfully");
}

TEST_F(ASTTranslatorFixedTest, HandlesNullPlanTree) {
    PGX_INFO("Testing null plan tree handling");
    
    // Create PlannedStmt with null plan tree
    PlannedStmt stmt{};
    memset(&stmt, 0, sizeof(PlannedStmt));
    stmt.type = T_PlannedStmt;
    stmt.commandType = 1;  // CMD_SELECT
    stmt.planTree = nullptr;
    
    // Translate - should handle null gracefully
    auto module = translator->translateQuery(&stmt);
    
    // Should fail gracefully
    ASSERT_EQ(module, nullptr) << "Translation should fail gracefully with null plan tree";
    
    PGX_INFO("Null plan tree handled gracefully");
}