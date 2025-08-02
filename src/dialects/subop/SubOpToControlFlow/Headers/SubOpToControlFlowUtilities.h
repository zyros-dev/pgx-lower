#pragma once

#include "SubOpToControlFlowCommon.h"

#include <vector>
#include <functional>
#include <unordered_map>
#include <string>
#include <optional>

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

} // namespace subop_to_control_flow

// Template implementation (must be in header)
namespace subop_to_control_flow {

template <class T>
std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

} // namespace subop_to_control_flow