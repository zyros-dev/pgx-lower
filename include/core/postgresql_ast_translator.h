#ifndef POSTGRESQL_AST_TRANSLATOR_H
#define POSTGRESQL_AST_TRANSLATOR_H

#include "mlir_logger.h"
#include <memory>

// Forward declarations to avoid include issues
namespace mlir {
class MLIRContext;
class ModuleOp;
class Value;
class OpBuilder;
class Operation;
class Type;
class Location;
}

// PostgreSQL C headers - need extern "C" wrapping
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
struct SeqScan;
struct List;
struct Node;
typedef unsigned int Oid;
}

namespace postgresql_ast {
class PostgreSQLASTTranslator {
public:
    explicit PostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger);
    ~PostgreSQLASTTranslator() = default;

    // Main translation entry points
    auto translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<mlir::ModuleOp>;
private:
    // Expression translation (recursive descent)
    auto translateExpression(Expr* expr) -> mlir::Value;
    auto translateOpExpr(OpExpr* opExpr) -> mlir::Value;
    auto translateVar(Var* var) -> mlir::Value;
    auto translateConst(Const* constNode) -> mlir::Value;
    auto translateFuncExpr(FuncExpr* funcExpr) -> mlir::Value;
    auto translateBoolExpr(BoolExpr* boolExpr) -> mlir::Value;
    auto translateNullTest(NullTest* nullTest) -> mlir::Value;
    auto translateAggref(Aggref* aggref) -> mlir::Value;
    auto translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> mlir::Value;
    
    // Plan node translation
    auto translateSeqScan(SeqScan* seqScan) -> mlir::Operation*;
    auto translateProjection(List* targetList) -> mlir::Operation*;
    auto translateSelection(List* qual) -> mlir::Operation*;
    
    // Tuple iteration and result processing
    auto generateTupleIterationLoop(mlir::OpBuilder& builder, mlir::Location location, 
                                   SeqScan* seqScan, List* targetList) -> void;
    auto generateAggregateLoop(mlir::OpBuilder& builder, mlir::Location location,
                              SeqScan* seqScan, List* targetList) -> void;
    auto getFieldValue64(mlir::OpBuilder& builder, mlir::Location location,
                         int32_t aggregateFieldIndex, uint32_t aggregateFieldType,
                         mlir::Type ptrType, mlir::Type i32Type) -> mlir::Value;
    auto processTargetListWithRealTuple(mlir::OpBuilder& builder, mlir::Location location,
                                       mlir::Value tupleHandle, List* targetList) -> void;

    mlir::MLIRContext& context_;
    MLIRLogger& logger_;
    mlir::OpBuilder* builder_;  // Current builder context
    mlir::ModuleOp* currentModule_;  // Current module being built
    mlir::Value* currentTupleHandle_;  // Current tuple handle for field access (nullptr if none)
    
    auto registerDialects() -> void;
    auto createRuntimeFunctionDeclarations(mlir::ModuleOp& module) -> void;
    auto getMLIRTypeForPostgreSQLType(Oid typeOid) -> mlir::Type;
    auto getOperatorName(Oid operatorOid) -> const char*;
    auto canHandleExpression(Node* expr) -> bool;
    auto logExpressionInfo(Node* expr, const char* context) -> void;
    
    // AST node type validation
    auto isArithmeticOperator(const char* opName) -> bool;
    auto isComparisonOperator(const char* opName) -> bool;
    auto isLogicalOperator(const char* opName) -> bool;
    auto isTextOperator(const char* opName) -> bool;
};

// Factory function
auto createPostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    -> std::unique_ptr<PostgreSQLASTTranslator>;

} // namespace postgresql_ast

#endif // POSTGRESQL_AST_TRANSLATOR_H