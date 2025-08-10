#ifndef MLIR_SUPPORT_PARSING_H
#define MLIR_SUPPORT_PARSING_H

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace support {
enum TimeUnit {
   SECOND,
   MILLI,
   MICRO,
   NANO
};

enum DataType {
   NA,
   BOOL,
   INT8, INT16, INT32, INT64,
   UINT8, UINT16, UINT32, UINT64,
   HALF_FLOAT, FLOAT, DOUBLE,
   STRING,
   DECIMAL128,
   DATE32, DATE64,
   TIMESTAMP,
   FIXED_SIZE_BINARY,
   INTERVAL_MONTHS, INTERVAL_DAY_TIME
};

std::pair<uint64_t, uint64_t> getDecimalScaleMultiplier(int32_t scale);
std::pair<uint64_t, uint64_t> parseDecimal(std::string str, int32_t scale);
std::variant<int64_t, double, std::string> parse(std::variant<int64_t, double, std::string> val, DataType type, uint32_t param1 = 0, uint32_t param2 = 0);

} // end namespace support

#endif // MLIR_SUPPORT_PARSING_H
