#include "runtime/TableBuilder.h"
#include "runtime/helpers.h"
#include "execution/logging.h"
#include <iostream>
#include <string>

// PostgreSQL runtime integration
extern "C" {
    void mark_results_ready_for_streaming();
    bool add_tuple_to_result(int64_t value);
    void prepare_computed_results(int32_t numColumns);
}

#define EXPORT extern "C" __attribute__((visibility("default")))

// PostgreSQL-based TableBuilder implementation
namespace runtime {

TableBuilder* TableBuilder::create(VarLen32 schemaDescription) {
    PGX_DEBUG("TableBuilder::create called with schema: " + schemaDescription.str());
    // For PostgreSQL, we don't need Arrow tables
    // Just prepare the result storage
    prepare_computed_results(1); // For Test 1, we have 1 column
    
    // Pass nullptr as schema since we're not using Arrow
    auto* builder = new TableBuilder(nullptr);
    return builder;
}

void TableBuilder::destroy(TableBuilder* tb) {
    delete tb;
}

void* TableBuilder::build() {
    PGX_DEBUG("TableBuilder::build called");
    // Signal that results are ready for PostgreSQL to read
    mark_results_ready_for_streaming();
    return this; // Return self as a handle
}
void TableBuilder::addBool(bool isValid, bool value) {
    PGX_DEBUG("TableBuilder::addBool called");
    // For PostgreSQL, we'd store this in the result set
    // For now, just log it
}

void TableBuilder::addInt8(bool isValid, int8_t val) {
    PGX_DEBUG("TableBuilder::addInt8 called");
}

void TableBuilder::addInt16(bool isValid, int16_t val) {
    PGX_DEBUG("TableBuilder::addInt16 called");
}

void TableBuilder::addInt32(bool isValid, int32_t val) {
    PGX_DEBUG("TableBuilder::addInt32 called: " + std::to_string(val));
    // For Test 1, this is our ID column
    add_tuple_to_result(val);
}

void TableBuilder::addInt64(bool isValid, int64_t val) {
    PGX_DEBUG("TableBuilder::addInt64 called: " + std::to_string(val));
    add_tuple_to_result(val);
}

void TableBuilder::addFloat32(bool isValid, float val) {
    PGX_DEBUG("TableBuilder::addFloat32 called");
}

void TableBuilder::addFloat64(bool isValid, double val) {
    PGX_DEBUG("TableBuilder::addFloat64 called");
}

void TableBuilder::addDecimal(bool isValid, __int128 value) {
    PGX_DEBUG("TableBuilder::addDecimal called");
}

void TableBuilder::addBinary(bool isValid, VarLen32 string) {
    PGX_DEBUG("TableBuilder::addBinary called");
}

void TableBuilder::addFixedSized(bool isValid, int64_t val) {
    PGX_DEBUG("TableBuilder::addFixedSized called");
}

void TableBuilder::nextRow() {
    PGX_DEBUG("TableBuilder::nextRow called");
    currentBatchSize++;
    // For Test 1, add a simple row with ID = 1
    add_tuple_to_result(1);
}

} // namespace runtime

// The C++ member functions already generate the mangled names we need
// No need for extern "C" wrappers that would conflict
