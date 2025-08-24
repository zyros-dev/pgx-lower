extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/nodeFuncs.h" // For exprType
#include "nodes/pg_list.h" // For list iteration macros
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "utils/memutils.h" // For MemoryContext management
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "access/table.h" // For table_open/table_close
#include "utils/rel.h" // For get_rel_name, get_attname

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

    return std::make_unique<::mlir::ModuleOp>(module);
}


auto PostgreSQLASTTranslator::Impl::generateRelAlgOperations(::mlir::func::FuncOp queryFunc,
                                                       PlannedStmt* plannedStmt,
                                                       TranslationContext& context) -> bool {
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

    PGX_INFO("Checking if translated operation has results");
    if (translatedOp->getNumResults() > 0) {
        PGX_INFO("Operation has " + std::to_string(translatedOp->getNumResults()) + " results");
        auto result = translatedOp->getResult(0);
        PGX_INFO("Got result from translated operation");

        PGX_INFO("Checking result type");
        if (result.getType().isa<mlir::relalg::TupleStreamType>()) {
            PGX_INFO("Result is TupleStreamType, creating MaterializeOp");
            createMaterializeOp(context, result);
        }
        else {
        }
    }
    else {
    }

    context.builder->create<mlir::func::ReturnOp>(context.builder->getUnknownLoc());

    return true;
}


auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast