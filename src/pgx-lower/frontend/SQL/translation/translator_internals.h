#pragma once

extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"

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
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Translation Context Types
// ===========================================================================

namespace pgx_lower::frontend::sql {

// Type aliases for column mapping clarity
using varno_t = int;
using varattno_t = int;
using table_t = std::string;
using column_t = std::string;
using ColumnMapping = std::map<std::pair<varno_t, varattno_t>, std::pair<table_t, column_t>>;

// Column information structure for schema discovery
struct ColumnInfo {
    std::string name;
    unsigned int typeOid; // PostgreSQL type OID
    int32_t typmod; // Type modifier
    bool nullable;

    ColumnInfo(std::string n, unsigned int t, int32_t m, bool null)
    : name(std::move(n))
    , typeOid(t)
    , typmod(m)
    , nullable(null) {}
};

// Translation context for managing state during AST translation
struct TranslationContext {
    PlannedStmt* currentStmt = nullptr;
    ::mlir::OpBuilder* builder = nullptr;
    std::unordered_map<unsigned int, ::mlir::Type> typeCache; // Oid -> Type mapping
    ::mlir::Value currentTuple = nullptr;
    ColumnMapping columnMappings; // Maps (varno, varattno) -> (table_name, column_name)
};

} // namespace pgx_lower::frontend::sql

namespace postgresql_ast {

using TranslationContext = pgx_lower::frontend::sql::TranslationContext;

// ===========================================================================
// PostgreSQLASTTranslator::Impl
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
    auto translateSeqScan(SeqScan* seqScan, TranslationContext& context) const -> ::mlir::Operation*;
    auto translateAgg(Agg* agg, TranslationContext& context) -> ::mlir::Operation*;
    auto translateSort(const Sort* sort, TranslationContext& context) -> ::mlir::Operation*;
    auto translateLimit(const Limit* limit, TranslationContext& context) -> ::mlir::Operation*;
    auto translateGather(const Gather* gather, TranslationContext& context) -> ::mlir::Operation*;

    // Query function generation
    static auto createQueryFunction(::mlir::OpBuilder& builder, const TranslationContext& context)
        -> ::mlir::func::FuncOp;
    auto
    generateRelAlgOperations(::mlir::func::FuncOp queryFunc, const PlannedStmt* plannedStmt, TranslationContext& context)
        -> bool;

    // Relational operation helpers
    auto applySelectionFromQual(::mlir::Operation* inputOp, const List* qual, const TranslationContext& context)
        -> ::mlir::Operation*;
    auto applyProjectionFromTargetList(::mlir::Operation* inputOp, List* targetList, TranslationContext& context)
        -> ::mlir::Operation*;

    // Validation and column processing
    static auto validatePlanTree(const Plan* planTree) -> bool;
    auto extractTargetListColumns(TranslationContext& context,
                                  std::vector<::mlir::Attribute>& columnRefAttrs,
                                  std::vector<::mlir::Attribute>& columnNameAttrs) const -> bool;
    auto processTargetEntry(TranslationContext& context,
                            const List* tlist,
                            int index,
                            std::vector<::mlir::Attribute>& columnRefAttrs,
                            std::vector<::mlir::Attribute>& columnNameAttrs) const -> bool;
    auto determineColumnType(const TranslationContext& context, Expr* expr) const -> ::mlir::Type;
    auto createMaterializeOp(TranslationContext& context, ::mlir::Value tupleStream) const -> ::mlir::Operation*;

    // Operation translation helpers
    auto extractOpExprOperands(OpExpr* opExpr, ::mlir::Value& lhs, ::mlir::Value& rhs) -> bool;
    auto translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;
    auto translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;

   private:
    // Helper to map PostgreSQL aggregate function OIDs to LingoDB function names
    static auto getAggregateFunctionName(Oid aggfnoid) -> const char*;

    ::mlir::MLIRContext& context_;
    ::mlir::OpBuilder* builder_;
    ::mlir::ModuleOp* currentModule_;
    ::mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;
};

// ===========================================================================
// PostgreSQLTypeMapper
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
// SchemaManager
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
auto getColumnNameFromSchema(const PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> std::string;
auto getTableOidFromRTE(PlannedStmt* currentPlannedStmt, int varno) -> Oid;
auto isColumnNullable(const PlannedStmt* plannedStmt, int rtindex, AttrNumber attnum) -> bool;

// Translation helper for constants
auto translateConst(Const* constNode, ::mlir::OpBuilder& builder, ::mlir::MLIRContext& context) -> ::mlir::Value;

// Get all columns from a table
auto getAllTableColumnsFromSchema(const PlannedStmt* currentPlannedStmt, int scanrelid)
    -> std::vector<pgx_lower::frontend::sql::ColumnInfo>;

} // namespace postgresql_ast