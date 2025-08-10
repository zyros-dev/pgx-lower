// Include PostgreSQL headers first with proper C linkage
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/nodeFuncs.h"  // For exprType
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
}

// Undefine PostgreSQL macros that conflict with LLVM
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "frontend/SQL/postgresql_ast_translator.h"
#include "execution/logging.h"
#include "runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace postgresql_ast {

// Translation context for managing state
struct TranslationContext {
    PlannedStmt* currentStmt = nullptr;
    ::mlir::OpBuilder* builder = nullptr;
    std::unordered_map<Oid, ::mlir::Type> typeCache;
};

// Simple PostgreSQL type mapper
class PostgreSQLTypeMapper {
public:
    explicit PostgreSQLTypeMapper(::mlir::MLIRContext& context) : context_(context) {}
    
    ::mlir::Type mapPostgreSQLType(Oid typeOid) {
        switch (typeOid) {
            case INT4OID:
                return mlir::IntegerType::get(&context_, 32);
            case INT8OID:
                return mlir::IntegerType::get(&context_, 64);
            case INT2OID:
                return mlir::IntegerType::get(&context_, 16);
            case FLOAT4OID:
                return mlir::Float32Type::get(&context_);
            case FLOAT8OID:
                return mlir::Float64Type::get(&context_);
            case BOOLOID:
                return mlir::IntegerType::get(&context_, 1);
            default:
                PGX_WARNING("Unknown PostgreSQL type OID: " + std::to_string(typeOid) + ", defaulting to i32");
                return mlir::IntegerType::get(&context_, 32);
        }
    }
    
private:
    ::mlir::MLIRContext& context_;
};

PostgreSQLASTTranslator::PostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    : context_(context), builder_(nullptr), currentModule_(nullptr), 
      currentTupleHandle_(nullptr), currentPlannedStmt_(nullptr), contextNeedsRecreation_(false) {
    PGX_DEBUG("PostgreSQLASTTranslator initialized with minimal implementation");
}

auto PostgreSQLASTTranslator::translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp> {
    PGX_INFO("Starting PostgreSQL AST translation for PlannedStmt");
    
    if (!plannedStmt) {
        PGX_ERROR("PlannedStmt is null");
        return nullptr;
    }
    
    // Create MLIR module and builder context
    auto module = ::mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    ::mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create translation context
    TranslationContext context;
    context.currentStmt = plannedStmt;
    context.builder = &builder;
    
    // Create query function with RelAlg operations
    auto queryFunc = createQueryFunction(builder, context);
    if (!queryFunc) {
        PGX_ERROR("Failed to create query function");
        return nullptr;
    }
    
    // Generate RelAlg operations inside the function
    if (!generateRelAlgOperations(queryFunc, plannedStmt, context)) {
        PGX_ERROR("Failed to generate RelAlg operations");
        return nullptr;
    }
    
    PGX_INFO("PostgreSQL AST translation completed successfully");
    return std::make_unique<::mlir::ModuleOp>(module);
}

auto PostgreSQLASTTranslator::translatePlanNode(Plan* plan, TranslationContext& context) -> ::mlir::Operation* {
    if (!plan) {
        PGX_ERROR("Plan node is null");
        return nullptr;
    }
    
    PGX_DEBUG("Translating plan node of type: " + std::to_string(plan->type));
    
    switch (plan->type) {
        case T_SeqScan:
            return translateSeqScan(reinterpret_cast<SeqScan*>(plan), context);
        default:
            PGX_ERROR("Unsupported plan node type: " + std::to_string(plan->type));
            return nullptr;
    }
}

auto PostgreSQLASTTranslator::translateSeqScan(SeqScan* seqScan, TranslationContext& context) -> ::mlir::Operation* {
    if (!seqScan || !context.builder || !context.currentStmt) {
        PGX_ERROR("Invalid SeqScan parameters");
        return nullptr;
    }
    
    PGX_DEBUG("Translating SeqScan operation");
    
#ifdef POSTGRESQL_EXTENSION
    // Extract table information (only available in PostgreSQL extension environment)
    const auto rte = static_cast<RangeTblEntry*>(
        list_nth(context.currentStmt->rtable, seqScan->scan.scanrelid - 1));
    Oid tableOid = rte->relid;
    std::string tableName = get_rel_name(tableOid);
    std::string tableIdentifier = tableName + "|oid:" + std::to_string(tableOid);
#else
    // Mock values for unit test environment
    std::string tableName = "test";
    Oid tableOid = 16384;
    std::string tableIdentifier = tableName + "|oid:" + std::to_string(tableOid);
#endif
    
    PGX_DEBUG("Creating BaseTableOp for table: " + tableIdentifier);
    
    // Get tuple stream type
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context_);
    
    // Create BaseTableOp - simplified without region for Test 1
    auto baseTableOp = context.builder->create<pgx::mlir::relalg::BaseTableOp>(
        context.builder->getUnknownLoc(),
        tupleStreamType,
        context.builder->getStringAttr(tableIdentifier),
        context.builder->getI64IntegerAttr(tableOid)  // Use actual table OID
    );
    
    PGX_DEBUG("SeqScan translation completed successfully");
    return baseTableOp;
}

auto PostgreSQLASTTranslator::createQueryFunction(::mlir::OpBuilder& builder, TranslationContext& context) -> ::mlir::func::FuncOp {
    PGX_DEBUG("Creating query function using func::FuncOp pattern");
    
    // Get RelAlg Table type for return value - MaterializeOp produces !relalg.table
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context_);
    
    // Create func::FuncOp following LingoDB's pattern: func.func @query() -> !relalg.table
    auto queryFuncType = builder.getFunctionType({}, {relAlgTableType});
    auto queryFunc = builder.create<::mlir::func::FuncOp>(
        builder.getUnknownLoc(), "query", queryFuncType);
    
    // Create function body
    auto& queryBody = queryFunc.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&queryBody);
    
    return queryFunc;
}

auto PostgreSQLASTTranslator::generateRelAlgOperations(::mlir::func::FuncOp queryFunc, PlannedStmt* plannedStmt, TranslationContext& context) -> bool {
    PGX_DEBUG("Generating RelAlg operations inside function body");
    
    // Safety check for mock PlannedStmt in unit tests
    if (!plannedStmt->planTree) {
        PGX_ERROR("PlannedStmt planTree is null - likely mock data in unit test");
        return false;
    }
    
    // Translate the plan tree inside the function body
    auto baseTableOp = translatePlanNode(plannedStmt->planTree, context);
    if (!baseTableOp) {
        PGX_ERROR("Failed to translate plan node");
        return false;
    }
    
    // Get RelAlg Table type for MaterializeOp
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context_);
    
    // Materialize tuple stream to table using MaterializeOp
    // For SELECT *, we need to specify which columns to materialize
    // TODO Phase 4: Extract actual column list from SELECT statement
    llvm::SmallVector<::mlir::Attribute> columnAttrs;
    columnAttrs.push_back(context.builder->getStringAttr("*"));  // Placeholder for SELECT *
    auto columnsArrayAttr = context.builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = context.builder->create<pgx::mlir::relalg::MaterializeOp>(
        context.builder->getUnknownLoc(), relAlgTableType, baseTableOp->getResult(0), columnsArrayAttr);
    
    // Use standard func.return with materialized result
    context.builder->create<mlir::func::ReturnOp>(
        context.builder->getUnknownLoc(), materializeOp.getResult());
    
    PGX_DEBUG("RelAlg operations generated successfully");
    return true;
}

auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context);
}

} // namespace postgresql_ast