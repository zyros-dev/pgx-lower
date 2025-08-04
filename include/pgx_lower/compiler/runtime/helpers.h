#ifndef COMPILER_RUNTIME_HELPERS_H
#define COMPILER_RUNTIME_HELPERS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "compiler/Dialect/util/FunctionHelper.h"
#include <vector>
#include <string>

namespace pgx_lower::compiler::runtime {

// Base class for runtime function wrappers that generate MLIR calls
class RuntimeCallGenerator {
    mlir::OpBuilder& builder;
    mlir::Location loc;
    std::string functionName;
    
public:
    RuntimeCallGenerator(const std::string& name, mlir::OpBuilder& b, mlir::Location l) 
        : builder(b), loc(l), functionName(name) {}
    
    // Call operator to generate MLIR function calls
    std::vector<mlir::Value> operator()(std::initializer_list<mlir::Value> args) {
        return (*this)(std::vector<mlir::Value>(args));
    }
    
    std::vector<mlir::Value> operator()(mlir::ValueRange args) {
        return (*this)(std::vector<mlir::Value>(args.begin(), args.end()));
    }
    
    std::vector<mlir::Value> operator()(const std::vector<mlir::Value>& args);
};

// ExecutionContext runtime wrappers
struct ExecutionContext {
    static RuntimeCallGenerator allocStateRaw(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_exec_alloc_state_raw", builder, loc);
    }
    
    static RuntimeCallGenerator setTupleCount(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_exec_set_tuple_count", builder, loc);
    }
    
    static RuntimeCallGenerator setResult(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_exec_set_result", builder, loc);
    }
};

// ThreadLocal runtime wrappers
struct ThreadLocal {
    static RuntimeCallGenerator getLocal(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_threadlocal_get", builder, loc);
    }
    
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_threadlocal_create", builder, loc);
    }
    
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_threadlocal_merge", builder, loc);
    }
};

// DataSource runtime wrappers
struct DataSource {
    static RuntimeCallGenerator get(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_datasource_get", builder, loc);
    }
};

// DataSourceIteration runtime wrappers
struct DataSourceIteration {
    static RuntimeCallGenerator init(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_datasource_iteration_init", builder, loc);
    }
    
    static RuntimeCallGenerator iterate(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_datasource_iteration_iterate", builder, loc);
    }
};

// Buffer runtime wrappers
struct Buffer {
    static RuntimeCallGenerator createZeroed(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_buffer_create_zeroed", builder, loc);
    }
    
    static RuntimeCallGenerator iterate(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_buffer_iterate", builder, loc);
    }
};

// GrowingBuffer runtime wrappers
struct GrowingBuffer {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_create", builder, loc);
    }
    
    static RuntimeCallGenerator createIterator(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_create_iterator", builder, loc);
    }
    
    static RuntimeCallGenerator insert(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_insert", builder, loc);
    }
    
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_merge", builder, loc);
    }
    
    static RuntimeCallGenerator sort(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_sort", builder, loc);
    }
    
    static RuntimeCallGenerator asContinuous(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_as_continuous", builder, loc);
    }
};

} // namespace pgx_lower::compiler::runtime

// Include all runtime wrappers
#include "compiler/runtime/all_wrappers.h"

#endif // COMPILER_RUNTIME_HELPERS_H