#ifndef POSTGRESQL_AST_TRANSLATOR_H
#define POSTGRESQL_AST_TRANSLATOR_H

#include <memory>
#include <vector>

namespace mlir {
class Attribute;
class MLIRContext;
class ModuleOp;
class Value;
class OpBuilder;
class Operation;
class Type;
class Location;
namespace func {
class FuncOp;
}
}

extern "C" {
struct PlannedStmt;
struct SelectStmt;
struct Expr;
struct OpExpr;
struct Var;
struct Const;
struct FuncExpr;
struct BoolExpr;
struct NullTest;
struct Aggref;
struct CoalesceExpr;
struct TargetEntry;
struct SeqScan;
struct Agg;
struct Sort;
struct Limit;
struct Gather;
struct Plan;
struct List;
struct Node;
typedef unsigned int Oid;
typedef uintptr_t Datum;
}

namespace postgresql_ast {
class PostgreSQLASTTranslator {
public:
    struct ColumnInfo {
        std::string name;
        Oid typeOid;
        int32_t typmod;
        bool nullable;
    };

    explicit PostgreSQLASTTranslator(::mlir::MLIRContext& context);
    ~PostgreSQLASTTranslator() = default;

    auto translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp>;
private:
    auto translateExpression(Expr* expr) -> ::mlir::Value;
    auto translateOpExpr(OpExpr* opExpr) -> ::mlir::Value;
    auto translateVar(Var* var) -> ::mlir::Value;
    auto translateConst(Const* constNode) -> ::mlir::Value;
    auto translateFuncExpr(FuncExpr* funcExpr) -> ::mlir::Value;
    auto translateBoolExpr(BoolExpr* boolExpr) -> ::mlir::Value;
    auto translateNullTest(NullTest* nullTest) -> ::mlir::Value;
    auto translateAggref(Aggref* aggref) -> ::mlir::Value;
    auto translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> ::mlir::Value;
    
    auto translatePlanNode(Plan* plan, struct TranslationContext& context) -> ::mlir::Operation*;
    auto translateSeqScan(SeqScan* seqScan, struct TranslationContext& context) -> ::mlir::Operation*;
    auto translateAgg(Agg* agg, struct TranslationContext& context) -> ::mlir::Operation*;
    auto translateSort(Sort* sort, struct TranslationContext& context) -> ::mlir::Operation*;
    auto translateLimit(Limit* limit, struct TranslationContext& context) -> ::mlir::Operation*;
    auto translateGather(Gather* gather, struct TranslationContext& context) -> ::mlir::Operation*;
    
    auto createQueryFunction(::mlir::OpBuilder& builder, struct TranslationContext& context) -> ::mlir::func::FuncOp;
    auto generateRelAlgOperations(::mlir::func::FuncOp queryFunc, PlannedStmt* plannedStmt, struct TranslationContext& context) -> bool;
    
    auto applySelectionFromQual(::mlir::Operation* inputOp, List* qual, struct TranslationContext& context) -> ::mlir::Operation*;
    auto applyProjectionFromTargetList(::mlir::Operation* inputOp, List* targetList, struct TranslationContext& context) -> ::mlir::Operation*;
    
    auto validatePlanTree(Plan* planTree) -> bool;
    auto extractTargetListColumns(struct TranslationContext& context,
                                 std::vector<::mlir::Attribute>& columnRefAttrs,
                                 std::vector<::mlir::Attribute>& columnNameAttrs) -> bool;
    auto processTargetEntry(struct TranslationContext& context,
                           List* tlist,
                           int index,
                           std::vector<::mlir::Attribute>& columnRefAttrs,
                           std::vector<::mlir::Attribute>& columnNameAttrs) -> bool;
    auto determineColumnType(struct TranslationContext& context, Expr* expr) -> ::mlir::Type;
    auto createMaterializeOp(struct TranslationContext& context, ::mlir::Value tupleStream) -> ::mlir::Operation*;
    auto extractOpExprOperands(OpExpr* opExpr, ::mlir::Value& lhs, ::mlir::Value& rhs) -> bool;
    auto translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;
    auto translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value;

    ::mlir::MLIRContext& context_;
    ::mlir::OpBuilder* builder_;
    ::mlir::ModuleOp* currentModule_;
    ::mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;

    auto getTableNameFromRTE(int varno) -> std::string;
    auto getColumnNameFromSchema(int varno, int varattno) -> std::string;
    auto getAllTableColumnsFromSchema(int scanrelid) -> std::vector<ColumnInfo>;
};

auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    -> std::unique_ptr<PostgreSQLASTTranslator>;

}

#endif