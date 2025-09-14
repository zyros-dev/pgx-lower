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
    explicit PostgreSQLASTTranslator(::mlir::MLIRContext& context);
    ~PostgreSQLASTTranslator();

    auto translate_query(PlannedStmt* planned_stmt) const -> std::unique_ptr<::mlir::ModuleOp>;
    
private:

    class Impl;
    std::unique_ptr<Impl> p_impl_;
};

auto create_postgresql_ast_translator(::mlir::MLIRContext& context)
    -> std::unique_ptr<PostgreSQLASTTranslator>;

}

#endif