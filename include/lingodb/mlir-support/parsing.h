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
std::variant<int64_t, double, std::string> parse(std::variant<int64_t, double, std::string> val, int type, uint32_t param1 = 0, uint32_t param2 = 0);

} // end namespace support

#endif // MLIR_SUPPORT_PARSING_H
