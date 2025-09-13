#pragma once

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
#include "utils/syscache.h"
#include "access/htup_details.h"
#include "fmgr.h"

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
#include "pgx-lower/frontend/SQL/translation/translation_context.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"

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

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

namespace postgresql_ast {

using TranslationContext = pgx_lower::frontend::sql::TranslationContext;

// ===========================================================================
// PostgreSQLASTTranslator::Impl Class
// ===========================================================================
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

    // Plan node translation methods
    auto translatePlanNode(Plan* plan, TranslationContext& context) -> ::mlir::Operation*;
    auto translateSeqScan(SeqScan* seqScan, TranslationContext& context) -> ::mlir::Operation*;
    auto translateAgg(Agg* agg, TranslationContext& context) -> ::mlir::Operation*;
    auto translateSort(Sort* sort, TranslationContext& context) -> ::mlir::Operation*;
    auto translateLimit(Limit* limit, TranslationContext& context) -> ::mlir::Operation*;
    auto translateGather(Gather* gather, TranslationContext& context) -> ::mlir::Operation*;

    // Query function generation
    auto createQueryFunction(::mlir::OpBuilder& builder, TranslationContext& context) -> ::mlir::func::FuncOp;
    auto generateRelAlgOperations(::mlir::func::FuncOp queryFunc, PlannedStmt* plannedStmt, TranslationContext& context) -> bool;

    // Relational operation helpers
    auto applySelectionFromQual(::mlir::Operation* inputOp, List* qual, TranslationContext& context) -> ::mlir::Operation*;
    auto applyProjectionFromTargetList(::mlir::Operation* inputOp, List* targetList, TranslationContext& context) -> ::mlir::Operation*;

    // Validation and column processing
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

    // Operation translation helpers
    auto extractOpExprOperands(OpExpr* opExpr, ::mlir::Value& lhs, ::mlir::Value& rhs) -> bool;
    auto translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;
    auto translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;

private:
    // Helper to map PostgreSQL aggregate function OIDs to LingoDB function names
    auto getAggregateFunctionName(Oid aggfnoid) -> const char*;

    ::mlir::MLIRContext& context_;
    ::mlir::OpBuilder* builder_;
    ::mlir::ModuleOp* currentModule_;
    ::mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;
};

// ===========================================================================
// PostgreSQLTypeMapper Class
// ===========================================================================
class PostgreSQLTypeMapper {
public:
    explicit PostgreSQLTypeMapper(::mlir::MLIRContext& context)
        : context_(context) {}

    // Main type mapping function
    auto mapPostgreSQLType(Oid typeOid, int32_t typmod, bool nullable = false) -> ::mlir::Type;

    // Type modifier extraction functions
    auto extractNumericInfo(int32_t typmod) -> std::pair<int32_t, int32_t>;
    auto extractTimestampPrecision(int32_t typmod) -> mlir::db::TimeUnitAttr;
    auto extractVarcharLength(int32_t typmod) -> int32_t;

private:
    ::mlir::MLIRContext& context_;
};

// ===========================================================================
// SchemaManager Class
// ===========================================================================
class SchemaManager {
public:
    explicit SchemaManager(PlannedStmt* plannedStmt)
        : plannedStmt_(plannedStmt) {}

    // Table and column metadata access
    auto getTableNameFromRTE(int varno) -> std::string;
    auto getColumnNameFromSchema(int rtindex, AttrNumber attnum) -> std::string;
    auto getTableOidFromRTE(int varno) -> Oid;
    auto isColumnNullable(int rtindex, AttrNumber attnum) -> bool;

private:
    PlannedStmt* plannedStmt_;
};

// ===========================================================================
// Helper Functions
// ===========================================================================

// Aggregate function name mapping
auto getAggregateFunctionName(Oid aggfnoid) -> const char*;

// Schema access helpers (standalone functions for backwards compatibility)
auto getTableNameFromRTE(PlannedStmt* currentPlannedStmt, int varno) -> std::string;
auto getColumnNameFromSchema(PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> std::string;
auto getTableOidFromRTE(PlannedStmt* currentPlannedStmt, int varno) -> Oid;
auto isColumnNullable(PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> bool;

// Translation helper for constants
auto translateConst(Const* constNode, ::mlir::OpBuilder& builder, ::mlir::MLIRContext& context) -> ::mlir::Value;

// Get all columns from a table
auto getAllTableColumnsFromSchema(PlannedStmt* currentPlannedStmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast