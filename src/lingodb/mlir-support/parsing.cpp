#include "lingodb/mlir-support/parsing.h"

#include "pgx-lower/utility/logging.h"

#include <cmath>
#include <stdexcept>
#include <cstring>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
#include "fmgr.h"
}
#else
#define BOOLOID     16
#define CHAROID     18      // Single character type
#define INT8OID     20      // 64-bit integer (bigint)
#define INT2OID     21      // 16-bit integer (smallint)
#define INT4OID     23      // 32-bit integer (integer)
#define TEXTOID     25      // Text type
#define FLOAT4OID   700     // 32-bit float (real)
#define FLOAT8OID   701     // 64-bit float (double precision)
#define BPCHAROID   1042    // Blank-padded char(n)
#define VARCHAROID  1043    // Variable-length character
#define DATEOID     1082    // Date type
#define TIMESTAMPOID 1114   // Timestamp without timezone
#define INTERVALOID 1186    // Interval type
#define NUMERICOID  1700    // Numeric/Decimal type
#endif
int32_t parseDate32(std::string str) {
   if (str.length() != 10 || (str[4] != '-' && str[2] != '-')) {
       return 0; // Invalid format
   }

   int year, month, day;
   if (str[4] == '-') {
       // YYYY-MM-DD format
       year = std::stoi(str.substr(0, 4));
       month = std::stoi(str.substr(5, 2));
       day = std::stoi(str.substr(8, 2));
   } else {
       // MM-DD-YYYY format
       month = std::stoi(str.substr(0, 2));
       day = std::stoi(str.substr(3, 2));
       year = std::stoi(str.substr(6, 4));
   }

   // Calculate days since PostgreSQL epoch (2000-01-01)
   // Using standard Gregorian calendar algorithm
   int a = (14 - month) / 12;
   int y = year - a;
   int m = month + 12 * a - 3;
   int jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;

   // PostgreSQL epoch: 2000-01-01 = JDN 2451545
   int pg_epoch_jdn = 2451545;
   int days_since_pg_epoch = jdn - pg_epoch_jdn;

   return days_since_pg_epoch;
}
int convertTimeUnit(support::TimeUnit unit) {
   switch (unit) {
      case support::TimeUnit::SECOND: return 0;
      case support::TimeUnit::MILLI: return 1;
      case support::TimeUnit::MICRO: return 2;
      case support::TimeUnit::NANO: return 3;
   }
   return 0;
}

std::pair<uint64_t, uint64_t> support::getDecimalScaleMultiplier(const int32_t scale) {
    __uint128_t result = 1;
    result *= std::pow<__uint128_t>(static_cast<__uint128_t>(10), static_cast<__uint128_t>(scale));

    uint64_t low = static_cast<uint64_t>(result);
    uint64_t high = static_cast<uint64_t>(result >> 64);

    PGX_LOG(DB_LOWER, DEBUG, "getDecimalScaleMultiplier(scale=%d): low=0x%lx, high=0x%lx", scale, low, high);

    return {low, high};
}

std::pair<uint64_t, uint64_t> support::parseDecimal(std::string str, int32_t reqScale) {
   size_t dotPos = str.find('.');
   std::string integerPart = (dotPos != std::string::npos) ? str.substr(0, dotPos) : str;
   std::string fractionalPart = (dotPos != std::string::npos) ? str.substr(dotPos + 1) : "";

   bool negative = false;
   if (!integerPart.empty() && integerPart[0] == '-') {
       negative = true;
       integerPart = integerPart.substr(1);
   }

   while (fractionalPart.length() < reqScale) {
       fractionalPart += "0";
   }
   if (fractionalPart.length() > reqScale) {
       fractionalPart = fractionalPart.substr(0, reqScale);
   }

   std::string combinedStr = integerPart + fractionalPart;

   __int128 value = 0;
   for (char c : combinedStr) {
       if (c >= '0' && c <= '9') {
           value = value * 10 + (c - '0');
       }
   }

   if (negative) {
       value = -value;
   }

   uint64_t low = static_cast<uint64_t>(value & 0xFFFFFFFFFFFFFFFF);
   uint64_t high = static_cast<uint64_t>((value >> 64) & 0xFFFFFFFFFFFFFFFF);

   return {low, high};
}

std::variant<int64_t, double, std::string> parseInt(std::variant<int64_t, double, std::string> val) {
   int64_t res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      res = std::get<double>(val);
   } else {
      res = std::stoll(std::get<std::string>(val));
   }
   return res;
}
std::variant<int64_t, double, std::string> parseInterval(std::variant<int64_t, double, std::string> val) {
   int64_t res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse interval from double");
   } else {
      res = std::stoll(std::get<std::string>(val));
      if (std::get<std::string>(val).ends_with("days")) {
         res *= 24 * 60 * 60 * 1000000000ll;
      }
   }
   return res;
}
std::variant<int64_t, double, std::string> parseDouble(std::variant<int64_t, double, std::string> val) {
   double res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      res = std::get<double>(val);
   } else {
      res = std::stod(std::get<std::string>(val));
   }
   return res;
}
std::variant<int64_t, double, std::string> parseBool(std::variant<int64_t, double, std::string> val) {
   bool res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse bool from double");
   } else {
      auto str = std::get<std::string>(val);
      if (str == "true" || str == "t") {
         res = true;
      } else if (str == "false" || str == "f") {
         res = false;
      } else {
         throw std::runtime_error("can not parse bool from value: " + str);
      }
   }
   return static_cast<int64_t>(res);
}
std::variant<int64_t, double, std::string> parseString(std::variant<int64_t, double, std::string> val, bool acceptInts = false) {
   std::string str;
   if (std::holds_alternative<int64_t>(val)) {
      if (acceptInts) {
         str = std::to_string(std::get<int64_t>(val));
      } else {
         throw std::runtime_error("can not parse string from int: " + str);
      }
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse string from double: " + str);
   } else {
      str = std::get<std::string>(val);
   }
   return str;
}
std::variant<int64_t, double, std::string> parseDate(std::variant<int64_t, double, std::string> val, bool parse64 = false) {
   if (parse64) {
      // DateType<millisecond>: Convert to i64 milliseconds
      int64_t days;
      if (std::holds_alternative<int64_t>(val)) {
         days = std::get<int64_t>(val);
      } else if (std::holds_alternative<std::string>(val)) {
         days = parseDate32(std::get<std::string>(val));
      } else {
         throw std::runtime_error("can not parse date from double");
      }
      // Convert days to milliseconds
      return days * 24LL * 60LL * 60LL * 1000LL;
   } else {
      // DateType<day>: Keep as i32 days since PostgreSQL epoch (2000-01-01)
      int64_t days;
      if (std::holds_alternative<int64_t>(val)) {
         days = std::get<int64_t>(val);
      } else if (std::holds_alternative<std::string>(val)) {
         days = parseDate32(std::get<std::string>(val));
      } else {
         throw std::runtime_error("can not parse date from double");
      }
      return days;
   }
}
std::variant<int64_t, double, std::string> toI64(std::variant<int64_t, double, std::string> val) {
   if (std::holds_alternative<std::string>(val)) {
      int64_t res = 0;
      auto str = std::get<std::string>(val);
      memcpy(&res, str.data(), std::min(sizeof(res), str.size()));
      return res;
   }
   return val;
}
std::variant<int64_t, double, std::string> parseTimestamp(std::variant<int64_t, double, std::string> val, support::TimeUnit unit) {
   if (!std::holds_alternative<std::string>(val)) {
      throw std::runtime_error("can not parse timestamp");
   }
   std::string str = std::get<std::string>(val);

#ifdef POSTGRESQL_EXTENSION
   // Use PostgreSQL's built-in timestamp parsing
   extern Datum timestamp_in(PG_FUNCTION_ARGS);
   char* cstr = (char*)palloc(str.length() + 1);
   strcpy(cstr, str.c_str());
   Datum timestampDatum = DirectFunctionCall3(timestamp_in,
                                              CStringGetDatum(cstr),
                                              ObjectIdGetDatum(InvalidOid),
                                              Int32GetDatum(-1));
   Timestamp timestamp = DatumGetTimestamp(timestampDatum);

   // PostgreSQL timestamps are microseconds since 2000-01-01
   // Keep them in that format - don't convert to Unix epoch
   int64_t res = timestamp;

   pfree(cstr);
   return res;
#else
   return 0LL;
#endif
}
std::variant<int64_t, double, std::string> support::parse(std::variant<int64_t, double, std::string> val, int type, uint32_t param1, uint32_t param2) {
   // Use PostgreSQL type OIDs instead of Arrow types
    switch (type) {
    case INT2OID:
    case INT4OID:
    case INT8OID: return parseInt(val);
    case BOOLOID: return parseBool(val);
    case FLOAT4OID:
    case FLOAT8OID: return parseDouble(val);
    case NUMERICOID: return parseString(val, true);
    case TEXTOID: return parseString(val, true);
    case DATEOID: return parseDate(val, false);
    case TIMESTAMPOID: return parseTimestamp(val, static_cast<TimeUnit>(param1));
    case INTERVALOID: return parseInterval(val);
    default: PGX_ERROR("Failed to parse type"); throw std::runtime_error("can not parse type");
    }
}
