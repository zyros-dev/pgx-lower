#ifndef PGX_LOWER_RUNTIME_POSTGRESQLRUNTIME_H
#define PGX_LOWER_RUNTIME_POSTGRESQLRUNTIME_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include <vector>

namespace pgx_lower::compiler::runtime {

// PostgreSQL runtime functions that will replace Arrow functions
// These are stubs for now - will be implemented with actual PostgreSQL calls

class PostgreSQLTable {
public:
    static ::mlir::Value createEmpty(::mlir::OpBuilder& builder, ::mlir::Location loc);
    static ::mlir::Value addColumn(::mlir::OpBuilder& builder, ::mlir::Location loc, 
                                ::mlir::Value table, ::mlir::Value columnName, ::mlir::Value column);
};

class PostgreSQLColumnBuilder {
public:
    static ::mlir::Value create(::mlir::OpBuilder& builder, ::mlir::Location loc, ::mlir::Value typeDescr);
    static ::mlir::Value addBool(::mlir::OpBuilder& builder, ::mlir::Location loc, 
                              ::mlir::Value builder, ::mlir::Value isValid, ::mlir::Value val);
    static ::mlir::Value addFixedSized(::mlir::OpBuilder& builder, ::mlir::Location loc,
                                    ::mlir::Value builder, ::mlir::Value isValid, ::mlir::Value val);
    static ::mlir::Value addBinary(::mlir::OpBuilder& builder, ::mlir::Location loc,
                                ::mlir::Value builder, ::mlir::Value isValid, ::mlir::Value val);
    static ::mlir::Value merge(::mlir::OpBuilder& builder, ::mlir::Location loc,
                            ::mlir::Value left, ::mlir::Value right);
    static ::mlir::Value finish(::mlir::OpBuilder& builder, ::mlir::Location loc, ::mlir::Value builder);
};

// Temporary aliases to make existing code work
using ArrowTable = PostgreSQLTable;
using ArrowColumnBuilder = PostgreSQLColumnBuilder;

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLRUNTIME_H