#ifndef COMPILER_RUNTIME_EXECUTIONCONTEXT_H
#define COMPILER_RUNTIME_EXECUTIONCONTEXT_H

#include "pgx_lower/runtime/ExecutionContext.h"
#include "pgx_lower/runtime/RuntimeFunctions.h"

namespace pgx_lower::compiler::runtime {

// MLIR wrapper functions for ExecutionContext
struct ExecutionContext {
   static RuntimeFunction allocStateRaw(mlir::OpBuilder& builder, mlir::Location loc) {
      // TODO Phase 5: Implement ExecutionContext::allocStateRaw wrapper
      return RuntimeFunction("exec_alloc_state_raw", builder, loc);
   }
   
   static RuntimeFunction setTupleCount(mlir::OpBuilder& builder, mlir::Location loc) {
      // TODO Phase 5: Implement ExecutionContext::setTupleCount wrapper
      return RuntimeFunction("exec_set_tuple_count", builder, loc);
   }
   
   static RuntimeFunction setResult(mlir::OpBuilder& builder, mlir::Location loc) {
      // TODO Phase 5: Implement ExecutionContext::setResult wrapper
      return RuntimeFunction("exec_set_result", builder, loc);
   }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_EXECUTIONCONTEXT_H
