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

namespace pgx_lower { namespace runtime {

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

/**
 * Open a table for scanning
 * @param table_name Name of the table to scan
 * @return Opaque handle to the table scan, or nullptr on failure
 */
TableScanHandle* openTableScan(const char* table_name);

/**
 * Close a table scan and free resources
 * @param handle Table scan handle to close
 */
void closeTableScan(TableScanHandle* handle);

/**
 * Read the next tuple from a table scan
 * @param handle Table scan handle
 * @return Tuple handle, or nullptr if no more tuples
 */
TupleHandle* readNextTuple(TableScanHandle* handle);

//===----------------------------------------------------------------------===//
// Tuple Field Access
//===----------------------------------------------------------------------===//

/**
 * Extract integer field from tuple
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @param is_null Output parameter set to true if field is NULL
 * @return Field value (undefined if is_null is true)
 */
int32_t getIntField(TupleHandle* tuple, int field_index, bool* is_null);

/**
 * Extract 64-bit integer field from tuple
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @param is_null Output parameter set to true if field is NULL
 * @return Field value (undefined if is_null is true)
 */
int64_t getBigIntField(TupleHandle* tuple, int field_index, bool* is_null);

/**
 * Extract text field from tuple
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @param length Output parameter for text length
 * @param is_null Output parameter set to true if field is NULL
 * @return Pointer to text data (null-terminated, undefined if is_null is true)
 */
const char* getTextField(TupleHandle* tuple, int field_index, int* length, bool* is_null);

/**
 * Extract boolean field from tuple
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @param is_null Output parameter set to true if field is NULL
 * @return Field value (undefined if is_null is true)
 */
bool getBoolField(TupleHandle* tuple, int field_index, bool* is_null);

/**
 * Extract numeric field as double
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @param is_null Output parameter set to true if field is NULL
 * @return Field value as double (undefined if is_null is true)
 */
double getNumericField(TupleHandle* tuple, int field_index, bool* is_null);

//===----------------------------------------------------------------------===//
// Type Information
//===----------------------------------------------------------------------===//

/**
 * Get the PostgreSQL type OID for a field
 * @param tuple Tuple handle
 * @param field_index 0-based field index
 * @return PostgreSQL type OID
 */
uint32_t getFieldTypeOid(TupleHandle* tuple, int field_index);

/**
 * Get the number of fields in a tuple
 * @param tuple Tuple handle
 * @return Number of fields
 */
int getTupleFieldCount(TupleHandle* tuple);

//===----------------------------------------------------------------------===//
// Output Operations
//===----------------------------------------------------------------------===//

/**
 * Add tuple to query result set
 * @param tuple Tuple handle to output
 * @return true on success, false on failure
 */
bool outputTuple(TupleHandle* tuple);

/**
 * Create a new tuple with specified field values
 * @param field_count Number of fields
 * @param field_types Array of PostgreSQL type OIDs
 * @param field_values Array of field values (as int64_t for simplicity)
 * @param null_flags Array of null flags
 * @return New tuple handle, or nullptr on failure
 */
TupleHandle* createTuple(int field_count, uint32_t* field_types, int64_t* field_values, bool* null_flags);

/**
 * Free a tuple handle
 * @param tuple Tuple handle to free
 */
void freeTuple(TupleHandle* tuple);

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

/**
 * Compare two integer values with NULL handling
 * @param lhs Left operand
 * @param lhs_null True if left operand is NULL
 * @param rhs Right operand
 * @param rhs_null True if right operand is NULL
 * @param op Comparison operator ("eq", "ne", "lt", "le", "gt", "ge")
 * @return Comparison result (false if either operand is NULL)
 */
bool compareInt(int32_t lhs, bool lhs_null, int32_t rhs, bool rhs_null, const char* op);

/**
 * Compare two text values with NULL handling
 * @param lhs Left operand
 * @param lhs_null True if left operand is NULL
 * @param rhs Right operand
 * @param rhs_null True if right operand is NULL
 * @param op Comparison operator ("eq", "ne", "lt", "le", "gt", "ge")
 * @return Comparison result (false if either operand is NULL)
 */
bool compareText(const char* lhs, bool lhs_null, const char* rhs, bool rhs_null, const char* op);

//===----------------------------------------------------------------------===//
// Aggregation Support
//===----------------------------------------------------------------------===//

/**
 * Initialize sum aggregation state
 * @return Opaque aggregation state handle
 */
void* initSumAggregation();

/**
 * Add value to sum aggregation
 * @param state Aggregation state handle
 * @param value Value to add
 * @param is_null True if value is NULL
 */
void addToSum(void* state, int64_t value, bool is_null);

/**
 * Finalize sum aggregation and get result
 * @param state Aggregation state handle
 * @param result_null Output parameter set to true if result is NULL
 * @return Final sum value
 */
int64_t finalizeSumAggregation(void* state, bool* result_null);

/**
 * Free aggregation state
 * @param state Aggregation state handle to free
 */
void freeAggregationState(void* state);

}} // namespace pgx_lower::runtime

//===----------------------------------------------------------------------===//
// C-style interface for MLIR JIT compatibility
//===----------------------------------------------------------------------===//

extern "C" {

// Legacy function names for compatibility with existing MLIR code
void* open_postgres_table(const char* table_name);
void* read_next_tuple_from_table(void* table_handle);
void close_postgres_table(void* table_handle);
bool add_tuple_to_result(void* tuple_handle);

// Field access functions with MLIR-compatible signatures
int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null);
int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null); // returns char* as i64
}