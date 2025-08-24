#ifndef POSTGRESQL_AST_TRANSLATOR_H
#define POSTGRESQL_AST_TRANSLATOR_H

#include <memory>
#include <vector>
#include "pgx-lower/frontend/SQL/translation/translation_context.h"

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
    using ColumnInfo = pgx_lower::frontend::sql::ColumnInfo;

    explicit PostgreSQLASTTranslator(::mlir::MLIRContext& context);
    ~PostgreSQLASTTranslator();  // Must be defined in .cpp for Pimpl

    auto translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<::mlir::ModuleOp>;
    
private:

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

auto createPostgreSQLASTTranslator(::mlir::MLIRContext& context) 
    -> std::unique_ptr<PostgreSQLASTTranslator>;

}

#endif