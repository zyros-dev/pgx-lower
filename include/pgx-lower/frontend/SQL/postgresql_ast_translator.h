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
namespace relalg {
class ColumnDefAttr;
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
struct Result;
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

using PGResult = ::Result;  // PostgreSQL Result node type

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
    auto translateResult(PGResult* result, struct TranslationContext& context) -> ::mlir::Operation*;
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
    
    auto generateTupleIterationLoop(::mlir::OpBuilder& builder, ::mlir::Location location, 
                                   SeqScan* seqScan, List* targetList) -> void;
    auto generateAggregateLoop(::mlir::OpBuilder& builder, ::mlir::Location location,
                              SeqScan* seqScan, List* targetList) -> void;
    auto getFieldValue64(::mlir::OpBuilder& builder, ::mlir::Location location,
                         int32_t aggregateFieldIndex, uint32_t aggregateFieldType,
                         ::mlir::Type ptrType, ::mlir::Type i32Type) -> ::mlir::Value;
    auto processTargetListWithRealTuple(::mlir::OpBuilder& builder, ::mlir::Location location,
                                       ::mlir::Value tupleHandle, List* targetList) -> void;

    ::mlir::MLIRContext& context_;
    ::mlir::OpBuilder* builder_;
    ::mlir::ModuleOp* currentModule_;
    ::mlir::Value* currentTupleHandle_;
    PlannedStmt* currentPlannedStmt_;
    bool contextNeedsRecreation_;
    
    auto registerDialects() -> void;
    auto recreateContextAfterLoad() -> void;
    auto invalidateTypeCache() -> void;
    auto ensureContextIsolation() -> void;
    auto createRuntimeFunctionDeclarations(::mlir::ModuleOp& module) -> void;
    auto getMLIRTypeForPostgreSQLType(Oid typeOid) -> ::mlir::Type;
    auto getOperatorName(Oid operatorOid) -> const char*;
    auto canHandleExpression(Node* expr) -> bool;
    auto logExpressionInfo(Node* expr, const char* context) -> void;
    
    auto getTableNameFromRTE(int varno) -> std::string;
    auto getColumnNameFromSchema(int varno, int varattno) -> std::string;
    auto getAllTableColumnsFromSchema(int scanrelid) -> std::vector<ColumnInfo>;

    auto isArithmeticOperator(const char* opName) -> bool;
    auto isComparisonOperator(const char* opName) -> bool;
    auto isLogicalOperator(const char* opName) -> bool;
    auto isTextOperator(const char* opName) -> bool;
    
    auto generateDBDialectExpression(::mlir::OpBuilder& builder, ::mlir::Location location, 
                                    ::mlir::Value tupleArg, Expr* expr) -> ::mlir::Value;
    auto generateDBDialectOperand(::mlir::OpBuilder& builder, ::mlir::Location location,
                                 ::mlir::Value tupleArg, Node* operandNode) -> ::mlir::Value;
    auto generateDBConstant(::mlir::OpBuilder& builder, ::mlir::Location location,
                           Datum value, Oid typeOid, ::mlir::Type mlirType) -> ::mlir::Value;
    
    auto extractConstantAttr(::mlir::Value value, ::mlir::OpBuilder* builder) -> ::mlir::Attribute;
    auto createResultColumnDef(const std::string& name, ::mlir::Type type, int index) -> ::mlir::relalg::ColumnDefAttr;
    auto processResultTargetList(List* targetlist, TranslationContext& context,
                                 std::vector<::mlir::Attribute>& columnDefs,
                                 std::vector<::mlir::Attribute>& values) -> bool;
    auto translateFuncExprArgs(FuncExpr* funcExpr, std::vector<::mlir::Value>& args) -> bool;
    auto translateMathFunction(Oid funcId, const std::vector<::mlir::Value>& args) -> ::mlir::Value;
    auto translateStringFunction(Oid funcId, const std::vector<::mlir::Value>& args) -> ::mlir::Value;
    auto translateUnknownFunction(FuncExpr* funcExpr, const std::vector<::mlir::Value>& args) -> ::mlir::Value;
};

auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    -> std::unique_ptr<PostgreSQLASTTranslator>;

}

#endif