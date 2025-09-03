#include "lingodb/mlir-support/parsing.h"
#include <stdexcept>
#include <cstring>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
}
#else
#define BOOLOID     16
#define INT2OID     21
#define INT4OID     23
#define INT8OID     20
#define FLOAT4OID   700
#define FLOAT8OID   701
#define TEXTOID     25
#define NUMERICOID  1700
#define DATEOID     1082
#define TIMESTAMPOID 1114
#endif
int32_t parseDate32(std::string str) {
   // Simple date parsing - convert to days since epoch
   // Format: YYYY-MM-DD
   if (str.length() != 10 || str[4] != '-' || str[7] != '-') {
       return 0; // Invalid format
   }
   
   int year = std::stoi(str.substr(0, 4));
   int month = std::stoi(str.substr(5, 2));
   int day = std::stoi(str.substr(8, 2));
   
   // Simplified calculation: days since 1970-01-01
   // This is approximate - real PostgreSQL parsing would be more accurate
   int days = (year - 1970) * 365 + (month - 1) * 30 + day;
   return days;
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

std::pair<uint64_t, uint64_t> support::getDecimalScaleMultiplier(int32_t scale) {
   // Calculate scale multiplier for decimals (10^scale)
   uint64_t multiplier = 1;
   for (int i = 0; i < scale; i++) {
       multiplier *= 10;
   }
   // Return as high/low 64-bit parts (simplified - high part is 0)
   return {multiplier, 0};
}

std::pair<uint64_t, uint64_t> support::parseDecimal(std::string str, int32_t reqScale) {
   // Simple decimal parsing - convert string to integer representation
   double value = std::stod(str);
   
   // Scale the value 
   uint64_t multiplier = 1;
   for (int i = 0; i < reqScale; i++) {
       multiplier *= 10;
   }
   
   uint64_t scaled_value = (uint64_t)(value * multiplier);
   
   // Return as high/low 64-bit parts (simplified - high part is 0)
   return {scaled_value, 0};
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
   if (!std::holds_alternative<std::string>(val)) {
      throw std::runtime_error("can not parse date");
   }
   std::string str = std::get<std::string>(val);
   int64_t parsed = parseDate32(str);
   int64_t date64 = parsed * 24 * 60 * 60 * 1000000000ll;
   return date64;
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
   
   // Simple timestamp parsing - convert ISO format to microseconds since epoch
   // Format: YYYY-MM-DD HH:MM:SS
   // For now, just use a simple approximation
   int64_t res = 0;
   if (str.length() >= 19) {
       // Extract date part and convert to days
       std::string date_part = str.substr(0, 10);
       int32_t days = parseDate32(date_part);
       res = days * 24 * 60 * 60 * 1000000LL; // Convert to microseconds
       
       // Add time part (simplified)
       if (str.length() >= 19) {
           int hour = std::stoi(str.substr(11, 2));
           int minute = std::stoi(str.substr(14, 2)); 
           int second = std::stoi(str.substr(17, 2));
           res += (hour * 3600 + minute * 60 + second) * 1000000LL;
       }
   }
   
   return res;
}
std::variant<int64_t, double, std::string> support::parse(std::variant<int64_t, double, std::string> val, int type, uint32_t param1, uint32_t param2) {
   // Use PostgreSQL type OIDs instead of Arrow types
   switch (type) {
      case INT2OID:
      case INT4OID:
      case INT8OID:
         return parseInt(val);
      case BOOLOID: 
         return parseBool(val);
      case FLOAT4OID:
      case FLOAT8OID: 
         return parseDouble(val);
      case NUMERICOID: 
         return parseString(val, true);
      case TEXTOID: 
         return parseString(val, true);
      case DATEOID: 
         return parseDate(val, false);
      case TIMESTAMPOID: 
         return parseTimestamp(val, static_cast<TimeUnit>(param1));
      default:
         // For unknown types, try to parse as string
         return parseString(val, true);
   }
}
