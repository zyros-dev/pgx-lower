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

extern "C" {
        typedef int16_t int16;
    typedef int16 AttrNumber;
    typedef unsigned int Oid;
    
        struct Plan {
        int type;                       double startup_cost;            double total_cost;        
        double plan_rows;               int plan_width;                 
        bool parallel_aware;            bool parallel_safe;             bool async_capable;             int plan_node_id;               
        List* targetlist;               List* qual;                     Plan* lefttree;                 Plan* righttree;                List* initPlan;                 void* extParam;                 void* allParam;             };
    
    struct SeqScan {
        Plan plan;
        struct {
            int scanrelid;
        } scan;
    };
    
    struct Agg {
        Plan plan;
        int aggstrategy;              int aggsplit;                 int numCols;
        AttrNumber* grpColIdx;
        Oid* grpOperators;
        Oid* grpCollations;
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
        Node* limitOffset;            Node* limitCount;             int limitOption;              int uniqNumCols;          };
    
    struct Gather {
        Plan plan;
        int num_workers;
        int rescan_param;
        bool single_copy;
        bool invisible;
    };
    
    struct PlannedStmt {
        int type;                       int commandType;                uint64_t queryId;               bool hasReturning;              bool hasModifyingCTE;           bool canSetTag;                 bool transientPlan;             bool dependsOnRole;             bool parallelModeNeeded;         int jitFlags;                   Plan* planTree;                 List* rtable;               };
    
    typedef struct ListCell ListCell;
    
        struct List {
        int type;                   int length;                 int max_length;             ListCell* elements;         ListCell* initial_elements;     };
    
    struct ListCell {
        union {
            void* ptr_value;
            int int_value;
            unsigned int oid_value;
        } data;
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
    
        struct Var {
        Node node;
        uint32_t varno;                AttrNumber varattno;           Oid vartype;                  int32_t vartypmod;            Oid varcollid;                uint32_t varlevelsup;         uint32_t varnoold;            AttrNumber varoattno;         int location;             };
    
    struct OpExpr {
        Node node;
        Oid opno;                     Oid opfuncid;                 Oid opresulttype;             bool opretset;                Oid opcollid;                 Oid inputcollid;              List* args;                   int location;             };
    
    struct BoolExpr {
        Node node;
        int boolop;                   List* args;                   int location;             };
    
    struct TargetEntry {
        Node node;
        Node* expr;                   AttrNumber resno;             char* resname;                uint32_t ressortgroupref;         Oid resorigtbl;               AttrNumber resorigcol;         bool resjunk;             };
    
    struct FuncExpr {
        Node node;
        Oid funcid;                   Oid funcresulttype;           bool funcretset;              bool funcvariadic;            unsigned char funcformat;         Oid funccollid;               Oid inputcollid;              List* args;                   int location;             };
    
    struct Aggref {
        Node node;
        Oid aggfnoid;                 Oid aggtype;                  Oid aggcollid;                Oid inputcollid;              Oid aggtranstype;             List* aggargtypes;            List* aggdirectargs;          List* args;                   List* aggorder;               List* aggdistinct;            Node* aggfilter;              bool aggstar;                 bool aggvariadic;             char aggkind;                 uint32_t agglevelsup;         int location;             };
    
    struct NullTest {
        Node node;
        Node* arg;                    int nulltesttype;             bool argisrow;                int location;             };
    
    struct CoalesceExpr {
        Node node;
        Oid coalescetype;             Oid coalescecollid;   // Result collation OID
        List* args;           // List of expressions
        int location;             };
    
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
    #define T_Aggref 407
    #define T_NullTest 408
    #define T_CoalesceExpr 409
    
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
    
    // Null test types
    #define IS_NULL 0
    #define IS_NOT_NULL 1
    
    // Common aggregate function OIDs
    #define COUNT_STAR_OID 2147  // COUNT(*)
    
    // List access macros for PostgreSQL 17 compatibility
    // (These will be defined below after the NIL macro)
    #define SUM_INT4_OID 2108    // SUM(int4)
    #define AVG_INT4_OID 2101    // AVG(int4)
    #define MAX_INT4_OID 2116    // MAX(int4)
    #define MIN_INT4_OID 2132    // MIN(int4)
    
    // List manipulation macros for compatibility with PostgreSQL
    #define NIL ((List*)NULL)
    #define T_List 16  // NodeTag for List
    
    // Access list cell data (PostgreSQL 17 array-based lists)
    #define lfirst(lc) ((lc)->data.ptr_value)
    #define lfirst_int(lc) ((lc)->data.int_value)
    #define lfirst_oid(lc) ((lc)->data.oid_value)
    
    // Get list head (for array-based lists)
    static inline ListCell* list_head(const List* l) {
        return (l && l->length > 0) ? &l->elements[0] : NULL;
    }
    
    // Helper function to create a list with one element (PostgreSQL 17 style)
    static inline List* list_make1(void* x1) {
        List* list = new List{};
        list->type = T_List;
        list->length = 1;
        list->max_length = 4;  // Initial allocation
        
        // Allocate elements array
        list->elements = new ListCell[list->max_length];
        list->elements[0].data.ptr_value = x1;
        
        list->initial_elements = nullptr;  // Not using static allocation
        return list;
    }
    
    // Helper function to append to a list (PostgreSQL 17 style)
    static inline List* lappend(List* list, void* datum) {
        if (list == NIL) {
            return list_make1(datum);
        }
        
        // Check if we need to grow the array
        if (list->length >= list->max_length) {
            int new_max = list->max_length * 2;
            ListCell* new_elements = new ListCell[new_max];
            
            // Copy existing elements
            for (int i = 0; i < list->length; i++) {
                new_elements[i] = list->elements[i];
            }
            
            // Free old array if it wasn't the initial static allocation
            if (list->elements != list->initial_elements) {
                delete[] list->elements;
            }
            
            list->elements = new_elements;
            list->max_length = new_max;
        }
        
        // Add new element
        list->elements[list->length].data.ptr_value = datum;
        list->length++;
        
        return list;
    }
    
    // Helper function to create a list with two elements
    static inline List* list_make2(void* x1, void* x2) {
        List* list = list_make1(x1);
        return lappend(list, x2);
    }
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