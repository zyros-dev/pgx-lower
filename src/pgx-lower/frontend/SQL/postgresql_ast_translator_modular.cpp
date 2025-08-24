// Include PostgreSQL headers first with proper C linkage
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
}

// Undefine PostgreSQL macros that conflict with LLVM
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/translation/translation_context.h"
#include "pgx-lower/frontend/SQL/translation/expression_translator.h"
#include "pgx-lower/frontend/SQL/translation/plan_translator.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace postgresql_ast {

class PostgreSQLASTTranslator::Impl {
public:
    explicit Impl(::mlir::MLIRContext& context)
        : context_(context) {}
    
    auto translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp>;
    
private:
    // Core orchestration methods
    auto createQueryFunction(::mlir::OpBuilder& builder, pgx_lower::ast::TranslationContext& context) 
        -> ::mlir::func::FuncOp;
    auto generateRelAlgOperations(::mlir::func::FuncOp queryFunc, 
                                 PlannedStmt* plannedStmt, 
                                 pgx_lower::ast::TranslationContext& context) -> bool;
    auto validatePlanTree(Plan* planTree) -> bool;
    auto createMaterializeOp(pgx_lower::ast::TranslationContext& context, 
                           ::mlir::Value tupleStream) -> ::mlir::Operation*;
    auto extractTargetListColumns(pgx_lower::ast::TranslationContext& context,
                                 std::vector<::mlir::Attribute>& columnRefAttrs,
                                 std::vector<::mlir::Attribute>& columnNameAttrs) -> bool;
    
    ::mlir::MLIRContext& context_;
};

// Main translation entry point
auto PostgreSQLASTTranslator::Impl::translateQuery(PlannedStmt* plannedStmt) 
    -> std::unique_ptr<::mlir::ModuleOp> {
    
    if (!plannedStmt) {
        PGX_ERROR("PlannedStmt is null");
        return nullptr;
    }
    
    if (!plannedStmt->planTree) {
        PGX_ERROR("PlannedStmt has no plan tree");
        return nullptr;
    }
    
    // Create new module
    auto module = std::make_unique<::mlir::ModuleOp>(::mlir::ModuleOp::create(
        ::mlir::UnknownLoc::get(&context_)));
    
    // Create translation context
    pgx_lower::ast::TranslationContext trans_ctx;
    trans_ctx.mlir_context = &context_;
    trans_ctx.rtable = plannedStmt->rtable;
    trans_ctx.targetList = plannedStmt->planTree->targetlist;
    
    // Validate plan tree
    if (!validatePlanTree(plannedStmt->planTree)) {
        PGX_ERROR("Plan tree validation failed");
        return nullptr;
    }
    
    ::mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToEnd(module->getBody());
    trans_ctx.builder = &builder;
    
    // Create main query function
    auto queryFunc = createQueryFunction(builder, trans_ctx);
    if (!queryFunc) {
        PGX_ERROR("Failed to create query function");
        return nullptr;
    }
    
    trans_ctx.current_function = &queryFunc;
    
    // Generate RelAlg operations
    if (!generateRelAlgOperations(queryFunc, plannedStmt, trans_ctx)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        return nullptr;
    }
    
    return module;
}

auto PostgreSQLASTTranslator::Impl::createQueryFunction(
    ::mlir::OpBuilder& builder, 
    pgx_lower::ast::TranslationContext& context) -> ::mlir::func::FuncOp {
    
    auto loc = builder.getUnknownLoc();
    
    // Create function type: () -> ()
    auto funcType = builder.getFunctionType({}, {});
    
    // Create function
    auto funcOp = builder.create<::mlir::func::FuncOp>(loc, "query", funcType);
    funcOp.setVisibility(mlir::func::FuncOp::Visibility::Public);
    
    // Create entry block
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    context.current_block = entryBlock;
    
    return funcOp;
}

auto PostgreSQLASTTranslator::Impl::generateRelAlgOperations(
    ::mlir::func::FuncOp queryFunc,
    PlannedStmt* plannedStmt,
    pgx_lower::ast::TranslationContext& context) -> bool {
    
    if (!plannedStmt->planTree) {
        PGX_ERROR("No plan tree in PlannedStmt");
        return false;
    }
    
    auto loc = context.builder->getUnknownLoc();
    
    // Translate the plan tree using the plan translator
    auto* planOp = pgx_lower::ast::plan::translate(plannedStmt->planTree, context);
    if (!planOp) {
        PGX_ERROR("Failed to translate plan tree");
        return false;
    }
    
    // Get the tuple stream from the plan operation
    if (planOp->getNumResults() == 0) {
        PGX_ERROR("Plan operation has no results");
        return false;
    }
    
    auto tupleStream = planOp->getResult(0);
    
    // Create materialize operation
    auto* materializeOp = createMaterializeOp(context, tupleStream);
    if (!materializeOp) {
        PGX_ERROR("Failed to create materialize operation");
        return false;
    }
    
    // Create return
    context.builder->create<::mlir::func::ReturnOp>(loc);
    
    return true;
}

auto PostgreSQLASTTranslator::Impl::validatePlanTree(Plan* planTree) -> bool {
    if (!planTree) {
        PGX_ERROR("Plan tree is null");
        return false;
    }
    
    // Basic validation - check node type is valid
    switch (planTree->type) {
    case T_SeqScan:
    case T_Agg:
    case T_Sort:
    case T_Limit:
    case T_Gather:
        break;
    default:
        PGX_WARNING("Unknown plan node type: " + std::to_string(planTree->type));
        // Don't fail for unknown types, might be new PostgreSQL features
        break;
    }
    
    // Recursively validate child nodes
    if (planTree->lefttree && !validatePlanTree(planTree->lefttree)) {
        return false;
    }
    if (planTree->righttree && !validatePlanTree(planTree->righttree)) {
        return false;
    }
    
    return true;
}

auto PostgreSQLASTTranslator::Impl::createMaterializeOp(
    pgx_lower::ast::TranslationContext& context,
    ::mlir::Value tupleStream) -> ::mlir::Operation* {
    
    auto loc = context.builder->getUnknownLoc();
    
    // Extract column information for materialize
    std::vector<::mlir::Attribute> columnRefAttrs;
    std::vector<::mlir::Attribute> columnNameAttrs;
    
    if (!extractTargetListColumns(context, columnRefAttrs, columnNameAttrs)) {
        PGX_WARNING("Failed to extract target list columns, using defaults");
    }
    
    // Create arrays from column attributes
    auto columnRefs = context.builder->getArrayAttr(columnRefAttrs);
    auto columnNames = context.builder->getArrayAttr(columnNameAttrs);
    
    // Create table type for materialize
    auto tableType = mlir::dsa::TableType::get(context.mlir_context);
    
    // Create materialize operation
    auto materializeOp = context.builder->create<mlir::relalg::MaterializeOp>(
        loc, tableType, tupleStream, columnRefs, columnNames);
    
    return materializeOp;
}

auto PostgreSQLASTTranslator::Impl::extractTargetListColumns(
    pgx_lower::ast::TranslationContext& context,
    std::vector<::mlir::Attribute>& columnRefAttrs,
    std::vector<::mlir::Attribute>& columnNameAttrs) -> bool {
    
    if (!context.targetList) {
        return false;
    }
    
    ListCell* lc;
    foreach(lc, context.targetList) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle->resjunk) {
            // Create column reference
            std::string colName = tle->resname ? tle->resname : "col_" + std::to_string(tle->resno);
            
            auto* dialect = context.mlir_context->getOrLoadDialect<mlir::relalg::RelAlgDialect>();
            if (dialect) {
                auto& columnManager = dialect->getColumnManager();
                auto colRef = columnManager.createRef("result", colName);
                columnRefAttrs.push_back(colRef);
                columnNameAttrs.push_back(context.builder->getStringAttr(colName));
            }
        }
    }
    
    return !columnRefAttrs.empty();
}

// Public interface implementation
PostgreSQLASTTranslator::PostgreSQLASTTranslator(::mlir::MLIRContext& context)
    : pImpl(std::make_unique<Impl>(context)) {}

PostgreSQLASTTranslator::~PostgreSQLASTTranslator() = default;

auto PostgreSQLASTTranslator::translateQuery(PlannedStmt* plannedStmt) 
    -> std::unique_ptr<::mlir::ModuleOp> {
    return pImpl->translateQuery(plannedStmt);
}

auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast