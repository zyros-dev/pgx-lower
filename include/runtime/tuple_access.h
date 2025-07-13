#pragma once

#include <cstdint>
#include <string>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "access/htup_details.h"
#include "utils/lsyscache.h"
}
#endif

namespace pgx_lower::runtime {

/**
 * Runtime wrapper functions for PostgreSQL tuple access.
 * These provide a clean C++ interface that MLIR can call during JIT execution.
 */

// Forward declarations for opaque handles
struct TableScanHandle;
struct TupleHandle;

//===----------------------------------------------------------------------===//
// Table Operations
//===----------------------------------------------------------------------===//

auto openTableScan(const char* table_name) -> TableScanHandle*;

void closeTableScan(const TableScanHandle* handle);

auto readNextTuple(TableScanHandle* handle) -> TupleHandle*;

//===----------------------------------------------------------------------===//
// Tuple Field Access
//===----------------------------------------------------------------------===//

auto getIntField(TupleHandle* tuple, int field_index, bool* is_null) -> int32_t;

auto getBigIntField(TupleHandle* tuple, int field_index, bool* is_null) -> int64_t;

auto getTextField(TupleHandle* tuple, int field_index, int* length, bool* is_null) -> const char*;

auto getBoolField(TupleHandle* tuple, int field_index, bool* is_null) -> bool;

auto getNumericField(TupleHandle* tuple, int field_index, bool* is_null) -> double;

//===----------------------------------------------------------------------===//
// Type Information
//===----------------------------------------------------------------------===//

auto getFieldTypeOid(TupleHandle* tuple, int field_index) -> uint32_t;

auto getTupleFieldCount(TupleHandle* tuple) -> int;

//===----------------------------------------------------------------------===//
// Output Operations
//===----------------------------------------------------------------------===//

auto outputTuple(const TupleHandle* tuple) -> bool;

auto createTuple(int field_count, const uint32_t* field_types, const int64_t* field_values, const bool* null_flags) -> TupleHandle*;

void freeTuple(const TupleHandle* tuple);

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

auto compareInt(int32_t lhs, bool lhs_null, int32_t rhs, bool rhs_null, const char* op) -> bool;

auto compareText(const char* lhs, bool lhs_null, const char* rhs, bool rhs_null, const char* op) -> bool;

//===----------------------------------------------------------------------===//
// Aggregation Support
//===----------------------------------------------------------------------===//

auto initSumAggregation() -> void*;

void addToSum(void* state, int64_t value, bool is_null);

auto finalizeSumAggregation(void* state, bool* result_null) -> int64_t;

void freeAggregationState(void* state);

} // namespace pgx_lower::runtime

//===----------------------------------------------------------------------===//
// C-style interface for MLIR JIT compatibility
//===----------------------------------------------------------------------===//

extern "C" {

// Legacy function names for compatibility with existing MLIR code (commented out due to extern C conflicts)
// auto open_postgres_table(const char* table_name) -> void*;
// auto read_next_tuple_from_table(void* table_handle) -> void*;
// void close_postgres_table(void* table_handle);
// auto add_tuple_to_result(void* tuple_handle) -> bool;

// Field access functions with MLIR-compatible signatures
auto get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int32_t;
auto get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int64_t; // returns char* as i64
auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> double; // returns DECIMAL/NUMERIC as double
}