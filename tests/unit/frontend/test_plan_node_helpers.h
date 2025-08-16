#ifndef TEST_PLAN_NODE_HELPERS_H
#define TEST_PLAN_NODE_HELPERS_H

#include <gtest/gtest.h>
#include <cstddef>  // for offsetof
#include <cstring>  // for memset
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "pgx_lower/frontend/SQL/postgresql_ast_translator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSADialect.h"
#include "pgx_lower/mlir/Dialect/util/UtilDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/execution/logging.h"

// Mock PostgreSQL structures for testing
extern "C" {
    // Basic types
    typedef int16_t int16;
    typedef int16 AttrNumber;
    typedef unsigned int Oid;
    
    // Mock plan nodes for unit testing - must match PostgreSQL exactly!
    struct Plan {
        int type;               // NodeTag
        double startup_cost;    // Cost - estimated startup cost
        double total_cost;      // Cost - total cost  
        double plan_rows;       // estimated number of rows
        int plan_width;         // average row width in bytes
        
        // FIXED: Added fields that were missing
        bool parallel_aware;    // engage parallel-aware logic?
        bool parallel_safe;     // OK to use as part of parallel plan?
        bool async_capable;     // engage asynchronous-capable logic?
        int plan_node_id;       // unique across entire final plan tree
        
        List* targetlist;       // target list to be computed
        List* qual;             // qual conditions
        Plan* lefttree;         // left input plan tree
        Plan* righttree;        // right input plan tree
        List* initPlan;         // Init Plan nodes (uncorrelated subselects)
        void* extParam;         // external params affecting this node (Bitmapset*)
        void* allParam;         // all params affecting this node (Bitmapset*)
    };
    
    struct SeqScan {
        Plan plan;
        struct {
            int scanrelid;
        } scan;
    };
    
    struct Agg {
        Plan plan;
        int aggstrategy;      // AggStrategy enum
        int aggsplit;         // AggSplit enum - FIXED: was missing this field!
        int numCols;
        AttrNumber* grpColIdx;
        Oid* grpOperators;
        Oid* grpCollations;
        // Note: Additional fields exist in real PostgreSQL but not needed for tests
    };
    
    struct Sort {
        Plan plan;
        int numCols;
        AttrNumber* sortColIdx;
        Oid* sortOperators;
        Oid* collations;
        bool* nullsFirst;
    };
    
    struct Limit {
        Plan plan;
        Node* limitOffset;    // FIXED: Swapped order to match PostgreSQL
        Node* limitCount;     // OFFSET comes before COUNT in PostgreSQL
        int limitOption;      // FIXED: Added missing field
        int uniqNumCols;      // FIXED: Added missing field for LIMIT DISTINCT support
    };
    
    struct Gather {
        Plan plan;
        int num_workers;
        int rescan_param;
        bool single_copy;
        bool invisible;
    };
    
    struct PlannedStmt {
        int type;               // NodeTag - must be first!
        int commandType;        // CmdType enum
        uint64_t queryId;       // query identifier - FIXED: uint64_t to match PostgreSQL
        bool hasReturning;      // is it insert|update|delete RETURNING?
        bool hasModifyingCTE;   // has insert|update|delete in WITH?
        bool canSetTag;         // do I set the command result tag?
        bool transientPlan;     // is plan short-lived?
        bool dependsOnRole;     // needs to be replanned when role changes?
        bool parallelModeNeeded; // parallel mode needed?
        int jitFlags;           // JIT flags
        Plan* planTree;         // tree of Plan nodes
        List* rtable;           // list of RangeTblEntry nodes
    };
    
    struct List {
        void* head;
    };
    
    struct Node {
        int type;
    };
    
    struct RangeTblEntry {
        unsigned int relid;
    };
    
    struct Const {
        Node node;
        Oid consttype;
        int32_t consttypmod;
        Oid constcollid;
        int constlen;
        unsigned long constvalue;
        bool constisnull;
        bool constbyval;
        int location;
    };
    
    struct Param {
        Node node;
        int paramkind;
        int paramid;
        Oid paramtype;
        int32_t paramtypmod;
        Oid paramcollid;
        int location;
    };
    
    // Expression nodes for Tests 9-28
    struct Var {
        Node node;
        uint32_t varno;        // Index of this var's relation in the range table
        AttrNumber varattno;   // Attribute number of this var
        Oid vartype;          // Data type OID
        int32_t vartypmod;    // Type modifier
        Oid varcollid;        // Collation OID
        uint32_t varlevelsup; // For subqueries
        uint32_t varnoold;    // Original varno before query rewriting
        AttrNumber varoattno; // Original attribute number
        int location;         // Token location or -1
    };
    
    struct OpExpr {
        Node node;
        Oid opno;             // Operator OID
        Oid opfuncid;         // Underlying function OID
        Oid opresulttype;     // Result type OID
        bool opretset;        // True if operator returns a set
        Oid opcollid;         // Collation OID
        Oid inputcollid;      // Input collation OID
        List* args;           // Arguments to the operator (List of Expr)
        int location;         // Token location or -1
    };
    
    struct BoolExpr {
        Node node;
        int boolop;           // Type of boolean operator (AND, OR, NOT)
        List* args;           // Arguments (List of Expr)
        int location;         // Token location or -1
    };
    
    struct TargetEntry {
        Node node;
        Node* expr;           // Expression to compute or Var
        AttrNumber resno;     // Attribute number (1-based)
        char* resname;        // Name of the column (can be NULL)
        uint32_t ressortgroupref; // Nonzero if referenced by ORDER BY/GROUP BY
        Oid resorigtbl;       // OID of column's source table
        AttrNumber resorigcol; // Column's original attribute number
        bool resjunk;         // True if not a real output column
    };
    
    struct FuncExpr {
        Node node;
        Oid funcid;           // Function OID
        Oid funcresulttype;   // Result type OID
        bool funcretset;      // True if function returns a set
        bool funcvariadic;    // True if function uses VARIADIC
        unsigned char funcformat; // How to display the function call
        Oid funccollid;       // Collation OID for result
        Oid inputcollid;      // Input collation OID
        List* args;           // Arguments to the function
        int location;         // Token location or -1
    };
    
    // Plan node type constants
    #define T_PlannedStmt 67    // PlannedStmt node type
    #define T_SeqScan 335
    #define T_Agg 361
    #define T_Sort 358
    #define T_Limit 369
    #define T_Gather 364
    #define T_Const 400
    #define T_Param 401
    #define T_Var 402
    #define T_OpExpr 403
    #define T_BoolExpr 404
    #define T_TargetEntry 405
    #define T_FuncExpr 406
    
    // Aggregate strategy constants
    #define AGG_PLAIN 0
    #define AGG_SORTED 1
    #define AGG_HASHED 2
    #define AGG_MIXED 3
    
    // Boolean expression types
    #define AND_EXPR 0
    #define OR_EXPR 1
    #define NOT_EXPR 2
    
    // Common PostgreSQL OIDs for operators
    #define INT4EQOID 96      // = operator for int4
    #define INT4LTOID 97      // < operator for int4
    #define INT4GTOID 521     // > operator for int4
    #define INT4LEOID 523     // <= operator for int4
    #define INT4GEOID 525     // >= operator for int4
    #define INT4NEOID 518     // != operator for int4
    #define INT4PLUSOID 551   // + operator for int4
    #define INT4MINUSOID 552  // - operator for int4
    #define INT4MULOID 514    // * operator for int4
    #define INT4DIVOID 528    // / operator for int4
}

class PlanNodeTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize MLIR context with required dialects
        context = std::make_unique<mlir::MLIRContext>();
        
        // Load standard MLIR dialects needed by the translator
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        
        // Load custom pgx-lower dialects
        context->loadDialect<mlir::relalg::RelAlgDialect>();
        context->loadDialect<mlir::dsa::DSADialect>();
        context->loadDialect<mlir::util::UtilDialect>();
        context->loadDialect<mlir::db::DBDialect>();
        
        // Create translator
        translator = postgresql_ast::createPostgreSQLASTTranslator(*context);
    }
    
    // ===== Helper Functions for Node Creation =====
    
    // Create a SeqScan node with default values
    SeqScan* createSeqScan(int scanrelid = 1, double total_cost = 10.0, double plan_rows = 100) {
        SeqScan* seqScan = new SeqScan{};
        memset(seqScan, 0, sizeof(SeqScan));  // Clear all memory first
        seqScan->plan.type = T_SeqScan;
        seqScan->plan.startup_cost = 0.0;
        seqScan->plan.total_cost = total_cost;
        seqScan->plan.plan_rows = plan_rows;
        seqScan->plan.plan_width = 32;
        seqScan->plan.parallel_aware = false;
        seqScan->plan.parallel_safe = true;
        seqScan->plan.async_capable = false;
        seqScan->plan.plan_node_id = 0;
        seqScan->plan.targetlist = nullptr;
        seqScan->plan.qual = nullptr;
        seqScan->plan.lefttree = nullptr;
        seqScan->plan.righttree = nullptr;
        seqScan->scan.scanrelid = scanrelid;
        return seqScan;
    }
    
    // Create an Agg node with child and optional GROUP BY
    Agg* createAggNode(Plan* child, int aggstrategy, int numCols, AttrNumber* grpColIdx = nullptr,
                       double total_cost = 20.0, double plan_rows = 10) {
        Agg* agg = new Agg{};
        memset(agg, 0, sizeof(Agg));  // Clear all memory first
        agg->plan.type = T_Agg;
        agg->plan.startup_cost = 0.0;
        agg->plan.total_cost = total_cost;
        agg->plan.plan_rows = plan_rows;
        agg->plan.plan_width = 8;
        agg->plan.parallel_aware = false;
        agg->plan.parallel_safe = true;
        agg->plan.async_capable = false;
        agg->plan.plan_node_id = 0;
        agg->plan.targetlist = nullptr;
        agg->plan.qual = nullptr;
        agg->plan.lefttree = child;
        agg->plan.righttree = nullptr;
        agg->aggstrategy = aggstrategy;
        agg->aggsplit = 0;  // AGGSPLIT_SIMPLE - no split
        agg->numCols = numCols;
        agg->grpColIdx = grpColIdx;
        agg->grpOperators = nullptr;
        agg->grpCollations = nullptr;
        return agg;
    }
    
    // Create a Sort node with child and sort specifications
    Sort* createSortNode(Plan* child, int numCols, AttrNumber* sortColIdx,
                         Oid* sortOps = nullptr, bool* nullsFirst = nullptr,
                         double total_cost = 15.0, double plan_rows = 100) {
        Sort* sort = new Sort{};
        memset(sort, 0, sizeof(Sort));  // Ensure full initialization
        sort->plan.type = T_Sort;
        sort->plan.startup_cost = 0.0;
        sort->plan.total_cost = total_cost;
        sort->plan.plan_rows = plan_rows;
        sort->plan.plan_width = 32;
        sort->plan.parallel_aware = false;
        sort->plan.parallel_safe = true;
        sort->plan.async_capable = false;
        sort->plan.plan_node_id = 0;
        sort->plan.targetlist = nullptr;
        sort->plan.qual = nullptr;
        sort->plan.lefttree = child;
        sort->plan.righttree = nullptr;
        sort->numCols = numCols;
        sort->sortColIdx = sortColIdx;
        sort->sortOperators = sortOps;
        sort->nullsFirst = nullsFirst;
        sort->collations = nullptr;
        return sort;
    }
    
    // Create a Limit node with child and limit count
    Limit* createLimitNode(Plan* child, int limitCount,
                          double total_cost = 5.0, double plan_rows = -1) {
        Limit* limit = new Limit{};
        memset(limit, 0, sizeof(Limit));  // Clear all memory first
        limit->plan.type = T_Limit;
        limit->plan.startup_cost = 0.0;
        limit->plan.total_cost = total_cost;
        limit->plan.plan_rows = (plan_rows == -1) ? limitCount : plan_rows;
        limit->plan.plan_width = 32;
        limit->plan.parallel_aware = false;
        limit->plan.parallel_safe = true;
        limit->plan.async_capable = false;
        limit->plan.plan_node_id = 0;
        limit->plan.targetlist = nullptr;
        limit->plan.qual = nullptr;
        limit->plan.lefttree = child;
        limit->plan.righttree = nullptr;
        
        // Create Const node for limit count
        Const* limitConst = new Const{};
        limitConst->node.type = T_Const;
        limitConst->consttype = 23; // INT4OID
        limitConst->constvalue = limitCount;
        limitConst->constisnull = false;
        limitConst->constbyval = true;
        
        limit->limitOffset = nullptr;  // OFFSET comes first in PostgreSQL
        limit->limitCount = reinterpret_cast<Node*>(limitConst);
        limit->limitOption = 0;  // LIMIT_OPTION_COUNT - default option
        limit->uniqNumCols = 0;  // Not using LIMIT DISTINCT
        return limit;
    }
    
    // Create a Gather node with child
    Gather* createGatherNode(Plan* child, int num_workers = 2,
                            double total_cost = 25.0, double plan_rows = 10) {
        Gather* gather = new Gather{};
        memset(gather, 0, sizeof(Gather));  // Clear all memory first
        gather->plan.type = T_Gather;
        gather->plan.startup_cost = 0.0;
        gather->plan.total_cost = total_cost;
        gather->plan.plan_rows = plan_rows;
        gather->plan.plan_width = 8;
        gather->plan.parallel_aware = true;  // Gather is parallel-aware
        gather->plan.parallel_safe = true;
        gather->plan.async_capable = false;
        gather->plan.plan_node_id = 0;
        gather->plan.targetlist = nullptr;
        gather->plan.qual = nullptr;
        gather->plan.lefttree = child;
        gather->plan.righttree = nullptr;
        gather->num_workers = num_workers;
        gather->single_copy = false;
        gather->invisible = false;
        gather->rescan_param = -1;
        return gather;
    }
    
    // ===== Helper Functions for Expression Creation =====
    
    // Create a Var node
    Var* createVar(int varno, AttrNumber varattno, Oid vartype) {
        Var* var = new Var{};
        var->node.type = T_Var;
        var->varno = varno;
        var->varattno = varattno;
        var->vartype = vartype;
        var->vartypmod = -1;
        var->varcollid = 0;
        var->varlevelsup = 0;
        var->varnoold = varno;
        var->varoattno = varattno;
        var->location = -1;
        return var;
    }
    
    // Create a Const node
    Const* createConst(Oid consttype, long value) {
        Const* constNode = new Const{};
        constNode->node.type = T_Const;
        constNode->consttype = consttype;
        constNode->constvalue = value;
        constNode->constisnull = false;
        constNode->constbyval = true;
        constNode->consttypmod = -1;
        constNode->constcollid = 0;
        constNode->constlen = 4;
        constNode->location = -1;
        return constNode;
    }
    
    // Create an OpExpr node
    OpExpr* createOpExpr(Oid opno, List* args, Oid resulttype = 16) {
        OpExpr* opExpr = new OpExpr{};
        opExpr->node.type = T_OpExpr;
        opExpr->opno = opno;
        opExpr->opfuncid = opno;  // Simplified: using same ID
        opExpr->opresulttype = resulttype;
        opExpr->opretset = false;
        opExpr->opcollid = 0;
        opExpr->inputcollid = 0;
        opExpr->args = args;
        opExpr->location = -1;
        return opExpr;
    }
    
    // Create a BoolExpr node
    BoolExpr* createBoolExpr(int boolop, List* args) {
        BoolExpr* boolExpr = new BoolExpr{};
        boolExpr->node.type = T_BoolExpr;
        boolExpr->boolop = boolop;
        boolExpr->args = args;
        boolExpr->location = -1;
        return boolExpr;
    }
    
    // Create a TargetEntry node
    TargetEntry* createTargetEntry(Node* expr, AttrNumber resno, const char* resname,
                                   uint32_t ressortgroupref = 0, bool resjunk = false) {
        TargetEntry* entry = new TargetEntry{};
        entry->node.type = T_TargetEntry;
        entry->expr = expr;
        entry->resno = resno;
        entry->resname = const_cast<char*>(resname);
        entry->ressortgroupref = ressortgroupref;
        entry->resorigtbl = 0;
        entry->resorigcol = 0;
        entry->resjunk = resjunk;
        return entry;
    }
    
    // Create a FuncExpr node for aggregate functions
    FuncExpr* createFuncExpr(Oid funcid, Oid resulttype, List* args) {
        FuncExpr* funcExpr = new FuncExpr{};
        funcExpr->node.type = T_FuncExpr;
        funcExpr->funcid = funcid;
        funcExpr->funcresulttype = resulttype;
        funcExpr->funcretset = false;
        funcExpr->funcvariadic = false;
        funcExpr->funcformat = 0;
        funcExpr->funccollid = 0;
        funcExpr->inputcollid = 0;
        funcExpr->args = args;
        funcExpr->location = -1;
        return funcExpr;
    }
    
    // ===== Test Execution Helpers =====
    
    // Translate and validate a plan with basic checks
    void translateAndValidate(PlannedStmt* stmt, const std::vector<std::string>& expectedPatterns,
                             bool expectModule = true) {
        auto module = translator->translateQuery(stmt);
        
        if (expectModule) {
            ASSERT_NE(module, nullptr) << "Translation should produce a module";
            validateMLIR(module.get(), expectedPatterns);
        } else {
            ASSERT_EQ(module, nullptr) << "Translation should not produce a module";
        }
    }
    
    // Cleanup helper for dynamically allocated nodes
    ~PlanNodeTestBase() {
        // Note: In a real implementation, we'd want proper memory management
        // For tests, the OS will clean up when the process exits
    }
    
    // Helper function to validate MLIR output contains expected patterns
    void validateMLIR(mlir::ModuleOp* module, const std::vector<std::string>& expectedPatterns) {
        ASSERT_NE(module, nullptr) << "Module should not be null";
        
        // Dump the MLIR to string in proper textual format
        std::string actualMLIR;
        llvm::raw_string_ostream stream(actualMLIR);
        
        // Print the module - even though func dialect uses generic form,
        // we can still validate the structure
        module->print(stream);
        actualMLIR = stream.str();
        
        // Log expected patterns and actual MLIR
        PGX_INFO("=== EXPECTED MLIR PATTERNS ===");
        for (const auto& pattern : expectedPatterns) {
            PGX_INFO("  - Should contain: " + pattern);
        }
        
        PGX_INFO("=== ACTUAL MLIR OUTPUT ===");
        PGX_INFO(actualMLIR);
        PGX_INFO("=== END MLIR OUTPUT ===");
        
        // Validate each expected pattern is present
        for (const auto& pattern : expectedPatterns) {
            EXPECT_TRUE(actualMLIR.find(pattern) != std::string::npos) 
                << "Missing expected pattern: " << pattern;
        }
    }
    
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<postgresql_ast::PostgreSQLASTTranslator> translator;
    
    // Helper to properly initialize PlannedStmt with all required fields
    PlannedStmt createPlannedStmt(Plan* planTree) {
        PlannedStmt stmt{};
        stmt.type = 0;  // Set to 0 for unit tests (different from production T_PlannedStmt=326)
        stmt.commandType = 1;  // CMD_SELECT
        stmt.queryId = 0;
        stmt.hasReturning = false;
        stmt.hasModifyingCTE = false;
        stmt.canSetTag = true;
        stmt.transientPlan = false;
        stmt.dependsOnRole = false;
        stmt.parallelModeNeeded = false;
        stmt.jitFlags = 0;
        stmt.planTree = planTree;
        stmt.rtable = nullptr;
        return stmt;
    }
};

#endif // TEST_PLAN_NODE_HELPERS_H