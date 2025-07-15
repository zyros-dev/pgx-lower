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

extern "C" {
auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> double; // returns DECIMAL/NUMERIC as double
}