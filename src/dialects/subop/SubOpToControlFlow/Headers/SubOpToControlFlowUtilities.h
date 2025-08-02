#pragma once

#include "SubOpToControlFlowCommon.h"

#include <vector>
#include <functional>
#include <unordered_map>
#include <string>
#include <optional>
#include <type_traits>

using namespace mlir;

// Forward declarations
class ColumnMapping;

namespace pgx_lower::compiler::dialect::subop_to_cf {
    class SubOpRewriter;
}


namespace subop_to_control_flow {

// Namespace alias for convenience
namespace subop = pgx_lower::compiler::dialect::subop;

// Block inlining utilities
std::vector<mlir::Value> inlineBlock(mlir::Block* b, mlir::OpBuilder& rewriter, mlir::ValueRange arguments);

// Type conversion utilities
std::vector<Type> unpackTypes(mlir::ArrayAttr arr);
TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter);

// Hash table type helpers
mlir::TupleType getHtKVType(subop::HashMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHtKVType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter);
mlir::TupleType getHtKVType(subop::PreAggrHtType t, mlir::TypeConverter& converter);

mlir::TupleType getHtEntryType(subop::HashMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHtEntryType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter);
mlir::TupleType getHtEntryType(subop::PreAggrHtType t, mlir::TypeConverter& converter);

mlir::TupleType getHashMultiMapEntryType(subop::HashMultiMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHashMultiMapValueType(subop::HashMultiMapType t, mlir::TypeConverter& converter);

// Hash computation
mlir::Value hashKeys(std::vector<mlir::Value> keys, OpBuilder& rewriter, Location loc);

// Buffer iteration utilities
void implementBufferIterationRuntime(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::Value)> fn);
void implementBufferIteration(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::Value)> fn);

// Atomic operations utilities
bool checkAtomicStore(mlir::Operation* op);

// Template utilities
template <class T>
std::vector<T> repeat(T val, size_t times);

// Core terminator utilities - consolidated LingoDB-compatible functions
namespace TerminatorUtils {
    // Essential terminator management
    void ensureTerminator(mlir::Region& region, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureIfOpTermination(mlir::scf::IfOp ifOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureForOpTermination(mlir::scf::ForOp forOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureFunctionTermination(mlir::func::FuncOp funcOp, mlir::OpBuilder& rewriter);
    void createContextAppropriateTerminator(mlir::Block* block, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Validation utilities
    bool hasTerminator(mlir::Block& block);
    bool isValidTerminator(mlir::Operation* op);
    std::vector<mlir::Block*> findBlocksWithoutTerminators(mlir::Region& region);
    void reportTerminatorStatus(mlir::Operation* rootOp);
}

// Runtime call termination - focused on essential PostgreSQL and LingoDB patterns
namespace RuntimeCallTermination {
    // Core runtime call safety
    void ensurePostgreSQLCallTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureLingoDRuntimeCallTermination(mlir::Operation* runtimeCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Apply safety patterns
    void applyRuntimeCallSafetyToOperation(mlir::Operation* rootOp, mlir::OpBuilder& rewriter);
    
    // Critical function-specific patterns
    void ensureStoreIntResultTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Validation
    bool isPostgreSQLRuntimeCall(mlir::func::CallOp callOp);
    bool isLingoDRuntimeCall(mlir::Operation* op);
    
    // Comprehensive LingoDB pattern application
    void applyComprehensiveLingoDRuntimeTermination(mlir::Operation* rootOp, mlir::OpBuilder& rewriter);
}

} // namespace subop_to_control_flow

// Template implementation (must be in header)
namespace subop_to_control_flow {

template <class T>
std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

// Simplified template implementations - only essential functionality
namespace TerminatorUtils {

// Simple operation processing without complex template abstractions
template<typename OperationType>
void processOperationTerminators(OperationType op, mlir::OpBuilder& rewriter, mlir::Location loc) {
    for (auto& region : op->getRegions()) {
        ensureTerminator(region, rewriter, loc);
    }
}

} // namespace TerminatorUtils

} // namespace subop_to_control_flow