extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "access/table.h"
#include "utils/rel.h"
#include "utils/array.h"

typedef int16 AttrNumber;
typedef struct Bitmapset Bitmapset;

typedef struct Agg Agg;
typedef struct Sort Sort;
typedef struct Limit Limit;
typedef struct Gather Gather;

#define AGG_PLAIN 0
#define AGG_SORTED 1
#define AGG_HASHED 2
#define AGG_MIXED 3
}

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"
#include <cstddef> // for offsetof

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/Column.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/runtime/metadata.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "pgx-lower/frontend/SQL/translation/translation_context.h"

namespace postgresql_ast {

using TranslationContext = pgx_lower::frontend::sql::TranslationContext;

class PostgreSQLASTTranslator::Impl {
public:
    explicit Impl(::mlir::MLIRContext& context)
        : context_(context)
        , builder_(nullptr)
        , currentModule_(nullptr)
        , currentTupleHandle_(nullptr)
        , currentPlannedStmt_(nullptr)
        , contextNeedsRecreation_(false) {}
    
    auto translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp>;
    
    auto translateExpression(Expr* expr) -> ::mlir::Value;
    auto translateOpExpr(OpExpr* opExpr) -> ::mlir::Value;
    auto translateVar(Var* var) -> ::mlir::Value;
    auto translateConst(Const* constNode) -> ::mlir::Value;
    auto translateFuncExpr(FuncExpr* funcExpr) -> ::mlir::Value;
    auto translateBoolExpr(BoolExpr* boolExpr) -> ::mlir::Value;
    auto translateNullTest(NullTest* nullTest) -> ::mlir::Value;
    auto translateAggref(Aggref* aggref) -> ::mlir::Value;
    auto translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> ::mlir::Value;
    auto translateScalarArrayOpExpr(ScalarArrayOpExpr* scalarArrayOp) -> ::mlir::Value;
    auto translateCaseExpr(CaseExpr* caseExpr) -> ::mlir::Value;
    auto translateExpressionWithCaseTest(Expr* expr, ::mlir::Value caseTestValue) -> ::mlir::Value;
    
    auto translatePlanNode(Plan* plan, TranslationContext& context) -> ::mlir::Operation*;
    auto translateSeqScan(SeqScan* seqScan, TranslationContext& context) -> ::mlir::Operation*;
    auto translateAgg(Agg* agg, TranslationContext& context) -> ::mlir::Operation*;
    auto translateSort(Sort* sort, TranslationContext& context) -> ::mlir::Operation*;
    auto translateLimit(Limit* limit, TranslationContext& context) -> ::mlir::Operation*;
    auto translateGather(Gather* gather, TranslationContext& context) -> ::mlir::Operation*;
    
    auto createQueryFunction(::mlir::OpBuilder& builder, TranslationContext& context) -> ::mlir::func::FuncOp;
    auto generateRelAlgOperations(::mlir::func::FuncOp queryFunc, PlannedStmt* plannedStmt, TranslationContext& context) -> bool;
    
    auto applySelectionFromQual(::mlir::Operation* inputOp, List* qual, TranslationContext& context) -> ::mlir::Operation*;
    auto applyProjectionFromTargetList(::mlir::Operation* inputOp, List* targetList, TranslationContext& context) -> ::mlir::Operation*;
    
    auto validatePlanTree(Plan* planTree) -> bool;
    auto extractTargetListColumns(TranslationContext& context,
                                 std::vector<::mlir::Attribute>& columnRefAttrs,
                                 std::vector<::mlir::Attribute>& columnNameAttrs) -> bool;
    auto processTargetEntry(TranslationContext& context,
                           List* tlist,
                           int index,
                           std::vector<::mlir::Attribute>& columnRefAttrs,
                           std::vector<::mlir::Attribute>& columnNameAttrs) -> bool;
    auto determineColumnType(TranslationContext& context, Expr* expr) -> ::mlir::Type;
    auto createMaterializeOp(TranslationContext& context, ::mlir::Value tupleStream) -> ::mlir::Operation*;
    auto extractOpExprOperands(OpExpr* opExpr, ::mlir::Value& lhs, ::mlir::Value& rhs) -> bool;
    auto translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;
    auto translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;
    
private:
    // Helper to map PostgreSQL aggregate function OIDs to LingoDB function names
    auto getAggregateFunctionName(Oid aggfnoid) -> const char* {
        using namespace pgx_lower::frontend::sql::constants;
        
        switch (aggfnoid) {
            // SUM functions
            case PG_F_SUM_INT2:
            case PG_F_SUM_INT4:
            case PG_F_SUM_INT8:
            case PG_F_SUM_FLOAT4:
            case PG_F_SUM_FLOAT8:
            case PG_F_SUM_NUMERIC:
                return AGGREGATION_SUM_FUNCTION;
                
            // AVG functions
            case PG_F_AVG_INT2:
            case PG_F_AVG_INT4:
            case PG_F_AVG_INT8:
            case PG_F_AVG_FLOAT4:
            case PG_F_AVG_FLOAT8:
            case PG_F_AVG_NUMERIC:
                return AGGREGATION_AVG_FUNCTION;
                
            // COUNT functions
            case PG_F_COUNT_STAR:
            case PG_F_COUNT_ANY:
                return AGGREGATION_COUNT_FUNCTION;
                
            // MIN functions
            case PG_F_MIN_INT2:
            case PG_F_MIN_INT4:
            case PG_F_MIN_INT8:
            case PG_F_MIN_FLOAT4:
            case PG_F_MIN_FLOAT8:
            case PG_F_MIN_NUMERIC:
            case PG_F_MIN_TEXT:
                return AGGREGATION_MIN_FUNCTION;
                
            // MAX functions
            case PG_F_MAX_INT2:
            case PG_F_MAX_INT4:
            case PG_F_MAX_INT8:
            case PG_F_MAX_FLOAT4:
            case PG_F_MAX_FLOAT8:
            case PG_F_MAX_NUMERIC:
            case PG_F_MAX_TEXT:
                return AGGREGATION_MAX_FUNCTION;
                
            default:
                PGX_WARNING("Unknown aggregate function OID: %u, defaulting to count", aggfnoid);
                return AGGREGATION_COUNT_FUNCTION;
        }
    }

    ::mlir::MLIRContext& context_;
    ::mlir::OpBuilder* builder_;
    ::mlir::ModuleOp* currentModule_;
    ::mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;
};

// Include implementation files directly - no separate compilation units needed
namespace {
    #include "translation/translation_core.cpp"
    #include "translation/schema_manager.cpp"
    
    // Wrapper to call anonymous namespace functions  
    auto callTranslateConst(Const* constNode, ::mlir::OpBuilder& builder, ::mlir::MLIRContext& context) -> ::mlir::Value {
        return translateConst(constNode, builder, context);
    }
}

// Include expression translator module
#include "translation/expression_translator.cpp"

// Include plan translator module  
#include "translation/plan_translator.cpp"

PostgreSQLASTTranslator::PostgreSQLASTTranslator(::mlir::MLIRContext& context)
    : pImpl(std::make_unique<Impl>(context)) {}

PostgreSQLASTTranslator::~PostgreSQLASTTranslator() = default;

auto PostgreSQLASTTranslator::translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp> {
    return pImpl->translateQuery(plannedStmt);
}

auto PostgreSQLASTTranslator::Impl::translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp> {
    PGX_LOG(AST_TRANSLATE, IO, "translateQuery IN: PostgreSQL PlannedStmt (cmd=%d, canSetTag=%d)",
            plannedStmt ? plannedStmt->commandType : -1, plannedStmt ? plannedStmt->canSetTag : false);
    
    if (!plannedStmt) {
        PGX_ERROR("PlannedStmt is null");
        return nullptr;
    }

    auto module = ::mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    ::mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToStart(module.getBody());

    builder_ = &builder;
    currentPlannedStmt_ = plannedStmt;

    TranslationContext context;
    context.currentStmt = plannedStmt;
    context.builder = &builder;

    auto queryFunc = createQueryFunction(builder, context);
    if (!queryFunc) {
        PGX_ERROR("Failed to create query function");
        builder_ = nullptr;
        currentPlannedStmt_ = nullptr;
        return nullptr;
    }

    if (!generateRelAlgOperations(queryFunc, plannedStmt, context)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        builder_ = nullptr;
        currentPlannedStmt_ = nullptr;
        return nullptr;
    }

    builder_ = nullptr;
    currentPlannedStmt_ = nullptr;

    auto result = std::make_unique<::mlir::ModuleOp>(module);
    auto numOps = module.getBody()->getOperations().size();
    PGX_LOG(AST_TRANSLATE, IO, "translateQuery OUT: RelAlg MLIR Module with %zu operations", numOps);
    
    return result;
}


auto PostgreSQLASTTranslator::Impl::generateRelAlgOperations(::mlir::func::FuncOp queryFunc,
                                                       PlannedStmt* plannedStmt,
                                                       TranslationContext& context) -> bool {
    PGX_LOG(AST_TRANSLATE, IO, "generateRelAlgOperations IN: PlannedStmt with planTree type %d",
            plannedStmt ? (plannedStmt->planTree ? plannedStmt->planTree->type : -1) : -1);
    
    if (!plannedStmt) {
        PGX_ERROR("PlannedStmt is null");
        return false;
    }

    Plan* planTree = plannedStmt->planTree;

    if (!validatePlanTree(planTree)) {
        return false;
    }

    auto translatedOp = translatePlanNode(planTree, context);
    if (!translatedOp) {
        PGX_ERROR("Failed to translate plan node");
        return false;
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Checking if translated operation has results");
    if (translatedOp->getNumResults() > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Operation has %d results", translatedOp->getNumResults());
        auto result = translatedOp->getResult(0);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Got result from translated operation");

        PGX_LOG(AST_TRANSLATE, DEBUG, "Checking result type");
        if (result.getType().isa<mlir::relalg::TupleStreamType>()) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Result is TupleStreamType, creating MaterializeOp");
            createMaterializeOp(context, result);
        }
        else {
        }
    }
    else {
    }

    context.builder->create<mlir::func::ReturnOp>(context.builder->getUnknownLoc());

    PGX_LOG(AST_TRANSLATE, IO, "generateRelAlgOperations OUT: RelAlg operations generated successfully");
    
    return true;
}


auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast