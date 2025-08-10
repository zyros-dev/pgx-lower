#include "mlir-support/parsing.h"
#include <cstdint>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cassert>
int32_t parseDate32(std::string str) {
   // Simple date parsing - assumes YYYY-MM-DD format
   // Returns days since epoch (1970-01-01)
   std::istringstream ss(str);
   std::string token;
   
   std::getline(ss, token, '-');
   int year = std::stoi(token);
   std::getline(ss, token, '-');
   int month = std::stoi(token);
   std::getline(ss, token, '-');
   int day = std::stoi(token);
   
   // Simple calculation for days since epoch
   // This is a simplified version - for production use a proper date library
   int days = (year - 1970) * 365 + (year - 1969) / 4;
   
   // Add days for months (approximate)
   int monthDays[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
   if (month > 0 && month <= 12) {
       days += monthDays[month - 1];
   }
   days += day - 1;
   
   return days;
}
// Time unit conversion removed - no longer needed without Arrow

std::pair<uint64_t, uint64_t> support::getDecimalScaleMultiplier(int32_t scale) {
   // Simple power-of-10 calculation for decimal scaling
   // This replaces Arrow's Decimal128::GetScaleMultiplier
   if (scale <= 0) return {1, 0};
   
   uint64_t multiplier = 1;
   for (int i = 0; i < scale && i < 19; ++i) {  // Max ~19 digits for uint64_t
       multiplier *= 10;
   }
   return {multiplier, 0};  // For simplicity, we use only the low 64 bits
}
std::pair<uint64_t, uint64_t> support::parseDecimal(std::string str, int32_t reqScale) {
   // Simple decimal parsing without Arrow dependency
   // Format: "123.456" or "123"
   size_t dotPos = str.find('.');
   
   std::string intPart = (dotPos != std::string::npos) ? str.substr(0, dotPos) : str;
   std::string fracPart = (dotPos != std::string::npos) ? str.substr(dotPos + 1) : "";
   
   // Parse integer part
   int64_t intValue = 0;
   if (!intPart.empty()) {
       intValue = std::stoll(intPart);
   }
   
   // Handle fractional part and scaling
   uint64_t result = std::abs(intValue);
   
   // Scale up by required decimal places
   for (int i = 0; i < reqScale; ++i) {
       result *= 10;
   }
   
   // Add fractional part (scaled appropriately)
   if (!fracPart.empty()) {
       int64_t fracValue = std::stoll(fracPart);
       int fracScale = fracPart.length();
       
       // Adjust fractional value to match required scale
       if (fracScale < reqScale) {
           for (int i = fracScale; i < reqScale; ++i) {
               fracValue *= 10;
           }
       } else if (fracScale > reqScale) {
           for (int i = reqScale; i < fracScale; ++i) {
               fracValue /= 10;
           }
       }
       result += fracValue;
   }
   
   // Handle sign
   if (intValue < 0) {
       result = -result;
   }
   
   return {result, 0};  // For simplicity, using only low 64 bits
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
   
   // Simple timestamp parsing - assumes ISO format: "YYYY-MM-DD HH:MM:SS"
   // This is a simplified implementation replacing Arrow parsing
   // TODO: Implement proper timestamp parsing for production use
   
   // For now, just return the date part as milliseconds since epoch
   // Extract date part (before space if exists)
   size_t spacePos = str.find(' ');
   std::string datePart = (spacePos != std::string::npos) ? str.substr(0, spacePos) : str;
   
   int64_t daysSinceEpoch = parseDate32(datePart);
   
   // Convert to appropriate time unit
   int64_t result = daysSinceEpoch * 24 * 60 * 60;  // seconds
   switch (unit) {
       case support::TimeUnit::MILLI: result *= 1000; break;
       case support::TimeUnit::MICRO: result *= 1000000; break;  
       case support::TimeUnit::NANO: result *= 1000000000LL; break;
       default: break; // SECOND - no change needed
   }
   
   return result;
}
std::variant<int64_t, double, std::string> support::parse(std::variant<int64_t, double, std::string> val, support::DataType type, uint32_t param1, uint32_t param2) {
   switch (type) {
      case support::DataType::INT8:
      case support::DataType::INT16:
      case support::DataType::INT32:
      case support::DataType::INT64:
      case support::DataType::UINT8:
      case support::DataType::UINT16:
      case support::DataType::UINT32:
      case support::DataType::UINT64:
      case support::DataType::INTERVAL_DAY_TIME:
      case support::DataType::INTERVAL_MONTHS:
         return parseInterval(val);
      case support::DataType::BOOL: return parseBool(val);
      case support::DataType::HALF_FLOAT:
      case support::DataType::FLOAT:
      case support::DataType::DOUBLE: return parseDouble(val);
      case support::DataType::FIXED_SIZE_BINARY: return toI64(parseString(val));
      case support::DataType::DECIMAL128: return parseString(val, true);
      case support::DataType::STRING: return parseString(val,true);
      case support::DataType::DATE32: return parseDate(val, false);
      case support::DataType::DATE64: return parseDate(val, true);
      case support::DataType::TIMESTAMP: return parseTimestamp(val, static_cast<TimeUnit>(param1));
      default:
         throw std::runtime_error("could not parse");
   }
}
