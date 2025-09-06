/**
 * StringRuntime.cpp - String operations runtime for LingoDB/pgx-lower
 * 
 * IMPORTANT: DO NOT MODIFY THIS FILE WITHOUT CAREFUL CONSIDERATION
 * 
 * This file is a direct port from LingoDB with minimal modifications.
 * It should maintain compatibility with LingoDB's runtime architecture.
 * 
 * Export Policy:
 * - Only exports the same functions as LingoDB's original StringRuntime.cpp
 * - Currently exports only: rt_varlen_from_ptr, rt_varlen_to_ref
 * - String operations (like, startsWith, etc.) are NOT exported as extern "C"
 * - They work through C++ symbol resolution via --export-dynamic
 * 
 * If you need to modify this file, ensure you understand the implications
 * for the entire MLIR lowering pipeline and runtime function registry.
 */

#include "pgx-lower/runtime/StringRuntime.h"
#include "lingodb/runtime/helpers.h"

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cctype>

//taken from NoisePage
// src: https://github.com/cmu-db/noisepage/blob/c2635d3360dd24a9f7a094b4b8bcd131d99f2d4b/src/execution/sql/operators/like_operators.cpp
// (MIT License, Copyright (c) 2018 CMU Database Group)
#define NextByte(p, plen) ((p)++, (plen)--)
bool iterativeLike(const char* str, size_t strLen, const char* pattern, size_t patternLen, char escape) {
   const char *s = str, *p = pattern;
   std::size_t slen = strLen, plen = patternLen;

   for (; plen > 0 && slen > 0; NextByte(p, plen)) {
      if (*p == escape) {
         // Next pattern character must match exactly, whatever it is
         NextByte(p, plen);

         if (plen == 0 || *p != *s) {
            return false;
         }

         NextByte(s, slen);
      } else if (*p == '%') {
         // Any sequence of '%' wildcards can essentially be replaced by one '%'. Similarly, any
         // sequence of N '_'s will blindly consume N characters from the input string. Process the
         // pattern until we reach a non-wildcard character.
         NextByte(p, plen);
         while (plen > 0) {
            if (*p == '%') {
               NextByte(p, plen);
            } else if (*p == '_') {
               if (slen == 0) {
                  return false;
               }
               NextByte(s, slen);
               NextByte(p, plen);
            } else {
               break;
            }
         }

         // If we've reached the end of the pattern, the tail of the input string is accepted.
         if (plen == 0) {
            return true;
         }

         if (*p == escape) {
            NextByte(p, plen);
            if (plen == 0) {
               return false;
            }
         }

         while (slen > 0) {
            if (iterativeLike(s, slen, p, plen, escape)) {
               return true;
            }
            NextByte(s, slen);
         }
         // No match
         return false;
      } else if (*p == '_') {
         // '_' wildcard matches a single character in the input
         NextByte(s, slen);
      } else if (*p == *s) {
         // Exact character match
         NextByte(s, slen);
      } else {
         // Unmatched!
         return false;
      }
   }
   while (plen > 0 && *p == '%') {
      NextByte(p, plen);
   }
   return slen == 0 && plen == 0;
}
//end taken from noisepage

bool runtime::StringRuntime::like(runtime::VarLen32 str1, runtime::VarLen32 str2) {
   return iterativeLike((str1).data(), (str1).getLen(), (str2).data(), (str2).getLen(), '\\');
}
bool runtime::StringRuntime::endsWith(runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).ends_with(std::string_view(str2.data(), str2.getLen()));
}
bool runtime::StringRuntime::startsWith(runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).starts_with(std::string_view(str2.data(), str2.getLen()));
}

// Helper function to trim leading and trailing spaces
static void trim_string(const char* data, int32_t len, const char** trimmed_data, int32_t* trimmed_len) {
   int32_t start = 0, end = len - 1;
   while (start <= end && data[start] == ' ') {
      ++start;
   }
   while (end >= start && data[end] == ' ') {
      --end;
   }
   *trimmed_len = end - start + 1;
   *trimmed_data = data + start;
}

// String to int64_t conversion
int64_t runtime::StringRuntime::toInt(runtime::VarLen32 str) {
   char* data = str.data();
   int32_t len = str.getLen();
   
   // Trim leading and trailing spaces
   const char* trimmed_data;
   int32_t trimmed_len;
   trim_string(data, len, &trimmed_data, &trimmed_len);
   
   // Create null-terminated string for strtoll
   std::string temp_str(trimmed_data, trimmed_len);
   
   // Parse the integer
   char* endptr;
   errno = 0;
   int64_t val = strtoll(temp_str.c_str(), &endptr, 10);
   
   // Check for errors
   if (errno == ERANGE || endptr != temp_str.c_str() + temp_str.length()) {
      std::string err = "Failed to cast the string " + std::string(data, len) + " to int64_t";
      throw std::runtime_error(err);
   }
   
   return val;
}

// String to float conversion
float runtime::StringRuntime::toFloat32(runtime::VarLen32 str) {
   char* data = str.data();
   int32_t len = str.getLen();
   
   // Trim leading and trailing spaces
   const char* trimmed_data;
   int32_t trimmed_len;
   trim_string(data, len, &trimmed_data, &trimmed_len);
   
   // Create null-terminated string for strtof
   std::string temp_str(trimmed_data, trimmed_len);
   
   // Parse the float
   char* endptr;
   errno = 0;
   float val = strtof(temp_str.c_str(), &endptr);
   
   // Check for errors
   if (errno == ERANGE || endptr != temp_str.c_str() + temp_str.length()) {
      std::string err = "Failed to cast the string " + std::string(data, len) + " to float";
      throw std::runtime_error(err);
   }
   
   return val;
}

// String to double conversion
double runtime::StringRuntime::toFloat64(runtime::VarLen32 str) {
   char* data = str.data();
   int32_t len = str.getLen();
   
   // Trim leading and trailing spaces
   const char* trimmed_data;
   int32_t trimmed_len;
   trim_string(data, len, &trimmed_data, &trimmed_len);
   
   // Create null-terminated string for strtod
   std::string temp_str(trimmed_data, trimmed_len);
   
   // Parse the double
   char* endptr;
   errno = 0;
   double val = strtod(temp_str.c_str(), &endptr);
   
   // Check for errors
   if (errno == ERANGE || endptr != temp_str.c_str() + temp_str.length()) {
      std::string err = "Failed to cast the string " + std::string(data, len) + " to double";
      throw std::runtime_error(err);
   }
   
   return val;
}

// String to decimal conversion
__int128 runtime::StringRuntime::toDecimal(runtime::VarLen32 string, int32_t reqScale) {
   std::string str = string.str();
   
   // Find the decimal point
   size_t decimal_pos = str.find('.');
   
   // Remove the decimal point and count digits after it
   int32_t scale = 0;
   std::string digits_str;
   
   if (decimal_pos != std::string::npos) {
      // Count digits after decimal point
      scale = str.length() - decimal_pos - 1;
      // Build string without decimal point
      digits_str = str.substr(0, decimal_pos) + str.substr(decimal_pos + 1);
   } else {
      digits_str = str;
   }
   
   // Parse as integer
   __int128 value = 0;
   bool negative = false;
   size_t start = 0;
   
   if (!digits_str.empty() && digits_str[0] == '-') {
      negative = true;
      start = 1;
   } else if (!digits_str.empty() && digits_str[0] == '+') {
      start = 1;
   }
   
   for (size_t i = start; i < digits_str.length(); i++) {
      if (digits_str[i] < '0' || digits_str[i] > '9') {
         throw std::runtime_error("could not cast decimal");
      }
      value = value * 10 + (digits_str[i] - '0');
   }
   
   if (negative) {
      value = -value;
   }
   
   // Rescale to required scale
   while (scale < reqScale) {
      value *= 10;
      scale++;
   }
   while (scale > reqScale) {
      value /= 10;
      scale--;
   }
   
   return value;
}

// int64_t to string conversion
runtime::VarLen32 runtime::StringRuntime::fromInt(int64_t value) {
   // Use a buffer large enough for any int64_t
   char buffer[32];
   int len = snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value));
   
   if (len < 0 || len >= static_cast<int>(sizeof(buffer))) {
      throw std::runtime_error("Failed to convert int64_t to string");
   }
   
   uint8_t* data = new uint8_t[len];
   memcpy(data, buffer, len);
   return runtime::VarLen32(data, len);
}

// float to string conversion
runtime::VarLen32 runtime::StringRuntime::fromFloat32(float value) {
   // Use a buffer large enough for float representation
   char buffer[64];
   int len = snprintf(buffer, sizeof(buffer), "%.9g", value);
   
   if (len < 0 || len >= static_cast<int>(sizeof(buffer))) {
      throw std::runtime_error("Failed to convert float to string");
   }
   
   uint8_t* data = new uint8_t[len];
   memcpy(data, buffer, len);
   return runtime::VarLen32(data, len);
}

// double to string conversion
runtime::VarLen32 runtime::StringRuntime::fromFloat64(double value) {
   // Use a buffer large enough for double representation
   char buffer[64];
   int len = snprintf(buffer, sizeof(buffer), "%.17g", value);
   
   if (len < 0 || len >= static_cast<int>(sizeof(buffer))) {
      throw std::runtime_error("Failed to convert double to string");
   }
   
   uint8_t* data = new uint8_t[len];
   memcpy(data, buffer, len);
   return runtime::VarLen32(data, len);
}

// decimal to string conversion
runtime::VarLen32 runtime::StringRuntime::fromDecimal(__int128 val, int32_t scale) {
   // Handle negative values
   bool negative = val < 0;
   if (negative) {
      val = -val;
   }
   
   // Convert to string
   std::string result;
   if (val == 0) {
      // For zero, we just return "0" regardless of scale
      result = "0";
      size_t len = result.length();
      uint8_t* data = new uint8_t[len];
      memcpy(data, result.data(), len);
      return runtime::VarLen32(data, len);
   } else {
      while (val > 0) {
         result = char('0' + (val % 10)) + result;
         val /= 10;
      }
   }
   
   // Add decimal point if needed (but not for zero which was handled above)
   if (scale > 0) {
      // Pad with zeros if necessary
      while (static_cast<int32_t>(result.length()) <= scale) {
         result = "0" + result;
      }
      // Insert decimal point
      result.insert(result.length() - scale, ".");
   }
   
   // Add negative sign if needed
   if (negative) {
      result = "-" + result;
   }
   
   size_t len = result.length();
   uint8_t* data = new uint8_t[len];
   memcpy(data, result.data(), len);
   
   return runtime::VarLen32(data, len);
}

runtime::VarLen32 runtime::StringRuntime::fromChar(uint64_t val, size_t bytes) {
   char* data = new char[bytes];
   memcpy(data, &val, bytes);
   return runtime::VarLen32((uint8_t*) data, bytes);
}

#define STR_CMP(NAME, OP)                                                                                  \
   bool runtime::StringRuntime::compare##NAME(runtime::VarLen32 str1, runtime::VarLen32 str2) {            \
      return std::string_view(str1.data(), str1.getLen()) OP std::string_view(str2.data(), str2.getLen()); \
   }

STR_CMP(Lt, <)
STR_CMP(Lte, <=)
STR_CMP(Gt, >)
STR_CMP(Gte, >=)

bool runtime::StringRuntime::compareEq(runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (str1.getLen() != str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()) == std::string_view(str2.data(), str2.getLen());
}
bool runtime::StringRuntime::compareNEq(runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (str1.getLen() != str2.getLen()) return true;
   return std::string_view(str1.data(), str1.getLen()) != std::string_view(str2.data(), str2.getLen());
}
EXPORT runtime::VarLen32 rt_varlen_from_ptr(uint8_t* ptr, uint32_t len) {
   return runtime::VarLen32(ptr, len);
}

EXPORT char* rt_varlen_to_ref(runtime::VarLen32* varlen) {
   return varlen->data();
}
runtime::VarLen32 runtime::StringRuntime::substr(runtime::VarLen32 str, size_t from, size_t to) {
   from -= 1;
   if (from > to || str.getLen() < to) throw std::runtime_error("can not perform substring operation");
   return runtime::VarLen32(&str.getPtr()[from], to - from);
}

size_t runtime::StringRuntime::findMatch(VarLen32 str, VarLen32 needle, size_t start, size_t end) {
   constexpr size_t invalidPos = 0x8000000000000000;
   if (start >= invalidPos) return invalidPos;
   if (start + needle.getLen() > end) return invalidPos;
   size_t found = std::string_view(str.data(), str.getLen()).find(std::string_view(needle.data(), needle.getLen()), start);

   if (found == std::string::npos || found + needle.getLen() > end) return invalidPos;
   return found + needle.getLen();
}

// String concatenation operations
runtime::VarLen32 runtime::StringRuntime::concat(runtime::VarLen32 left, runtime::VarLen32 right) {
   // Allocate space for concatenated string
   uint32_t totalLen = left.getLen() + right.getLen();
   uint8_t* result = new uint8_t[totalLen];
   
   // Copy left string
   memcpy(result, left.getPtr(), left.getLen());
   // Copy right string
   memcpy(result + left.getLen(), right.getPtr(), right.getLen());
   
   return runtime::VarLen32(result, totalLen);
}

runtime::VarLen32 runtime::StringRuntime::concat3(runtime::VarLen32 a, runtime::VarLen32 b, runtime::VarLen32 c) {
   uint32_t totalLen = a.getLen() + b.getLen() + c.getLen();
   uint8_t* result = new uint8_t[totalLen];
   
   uint32_t offset = 0;
   memcpy(result + offset, a.getPtr(), a.getLen());
   offset += a.getLen();
   memcpy(result + offset, b.getPtr(), b.getLen());
   offset += b.getLen();
   memcpy(result + offset, c.getPtr(), c.getLen());
   
   return runtime::VarLen32(result, totalLen);
}

// Case conversion operations
runtime::VarLen32 runtime::StringRuntime::upper(runtime::VarLen32 str) {
   uint32_t len = str.getLen();
   uint8_t* result = new uint8_t[len];
   
   for (uint32_t i = 0; i < len; i++) {
      char ch = static_cast<char>(str.getPtr()[i]);
      result[i] = static_cast<uint8_t>(std::toupper(ch));
   }
   
   return runtime::VarLen32(result, len);
}

runtime::VarLen32 runtime::StringRuntime::lower(runtime::VarLen32 str) {
   uint32_t len = str.getLen();
   uint8_t* result = new uint8_t[len];
   
   for (uint32_t i = 0; i < len; i++) {
      char ch = static_cast<char>(str.getPtr()[i]);
      result[i] = static_cast<uint8_t>(std::tolower(ch));
   }
   
   return runtime::VarLen32(result, len);
}

// PostgreSQL-style SUBSTRING (FROM pos FOR length)
runtime::VarLen32 runtime::StringRuntime::substring(runtime::VarLen32 str, int32_t start, int32_t length) {
   // PostgreSQL uses 1-based indexing
   if (start <= 0) {
      start = 1;
   }
   
   // Convert to 0-based index
   int32_t startIdx = start - 1;
   int32_t strLen = static_cast<int32_t>(str.getLen());
   
   // Handle out of bounds
   if (startIdx >= strLen || length <= 0) {
      return runtime::VarLen32(new uint8_t[0], 0);
   }
   
   // Adjust length if it exceeds string length
   if (startIdx + length > strLen) {
      length = strLen - startIdx;
   }
   
   uint8_t* result = new uint8_t[length];
   memcpy(result, str.getPtr() + startIdx, length);
   
   return runtime::VarLen32(result, length);
}

// Length operations
int32_t runtime::StringRuntime::length(runtime::VarLen32 str) {
   return static_cast<int32_t>(str.getLen());
}

int32_t runtime::StringRuntime::charLength(runtime::VarLen32 str) {
   // For now, assume single-byte encoding (UTF-8 support would need more work)
   return static_cast<int32_t>(str.getLen());
}

// Trimming operations
runtime::VarLen32 runtime::StringRuntime::trim(runtime::VarLen32 str) {
   const uint8_t* data = str.getPtr();
   int32_t len = str.getLen();
   
   // Find first non-space
   int32_t start = 0;
   while (start < len && data[start] == ' ') {
      start++;
   }
   
   // Find last non-space
   int32_t end = len - 1;
   while (end >= start && data[end] == ' ') {
      end--;
   }
   
   int32_t trimmedLen = end - start + 1;
   if (trimmedLen <= 0) {
      return runtime::VarLen32(new uint8_t[0], 0);
   }
   
   uint8_t* result = new uint8_t[trimmedLen];
   memcpy(result, data + start, trimmedLen);
   
   return runtime::VarLen32(result, trimmedLen);
}

runtime::VarLen32 runtime::StringRuntime::ltrim(runtime::VarLen32 str) {
   const uint8_t* data = str.getPtr();
   int32_t len = str.getLen();
   
   // Find first non-space
   int32_t start = 0;
   while (start < len && data[start] == ' ') {
      start++;
   }
   
   int32_t trimmedLen = len - start;
   if (trimmedLen <= 0) {
      return runtime::VarLen32(new uint8_t[0], 0);
   }
   
   uint8_t* result = new uint8_t[trimmedLen];
   memcpy(result, data + start, trimmedLen);
   
   return runtime::VarLen32(result, trimmedLen);
}

runtime::VarLen32 runtime::StringRuntime::rtrim(runtime::VarLen32 str) {
   const uint8_t* data = str.getPtr();
   int32_t len = str.getLen();
   
   // Find last non-space
   int32_t end = len - 1;
   while (end >= 0 && data[end] == ' ') {
      end--;
   }
   
   int32_t trimmedLen = end + 1;
   if (trimmedLen <= 0) {
      return runtime::VarLen32(new uint8_t[0], 0);
   }
   
   uint8_t* result = new uint8_t[trimmedLen];
   memcpy(result, data, trimmedLen);
   
   return runtime::VarLen32(result, trimmedLen);
}

// Case-insensitive pattern matching
bool runtime::StringRuntime::ilike(runtime::VarLen32 str, runtime::VarLen32 pattern) {
   // Convert both strings to lowercase for comparison
   runtime::VarLen32 lowerStr = lower(str);
   runtime::VarLen32 lowerPattern = lower(pattern);
   
   // Use the existing like function with lowercase strings
   bool result = like(lowerStr, lowerPattern);
   
   // Clean up allocated memory
   delete[] lowerStr.getPtr();
   delete[] lowerPattern.getPtr();
   
   return result;
}

