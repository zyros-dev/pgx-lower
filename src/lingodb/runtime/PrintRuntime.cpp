#include "lingodb/runtime/PrintRuntime.h"
#include "pgx-lower/utility/logging.h"
#include <cstring>

void runtime::PrintRuntime::print(const char* txt) {
   PGX_LOG(RUNTIME, DEBUG, "%s", txt);
}

void runtime::PrintRuntime::printVal(void* ptr, int32_t len) {
   if (!ptr || len <= 0) {
      PGX_LOG(RUNTIME, DEBUG, "printVal: null or invalid (ptr=%p, len=%d)", ptr, len);
      return;
   }

   const uint8_t* bytes = static_cast<const uint8_t*>(ptr);
   char hexbuf[1024];
   int pos = 0;

   for (int32_t i = 0; i < len && pos < 1000; i++) {
      pos += snprintf(hexbuf + pos, sizeof(hexbuf) - pos, "%02x ", bytes[i]);
   }

   PGX_LOG(RUNTIME, DEBUG, "printVal[%d]: %s", len, hexbuf);
}

void runtime::PrintRuntime::printPtr(void* ptr, int32_t offset, int32_t len) {
   if (!ptr || len <= 0) {
      PGX_LOG(RUNTIME, DEBUG, "printPtr: null or invalid (ptr=%p, offset=%d, len=%d)", ptr, offset, len);
      return;
   }

   const uint8_t* bytes = static_cast<const uint8_t*>(ptr) + offset;
   char hexbuf[1024];
   int pos = 0;

   for (int32_t i = 0; i < len && pos < 1000; i++) {
      pos += snprintf(hexbuf + pos, sizeof(hexbuf) - pos, "%02x ", bytes[i]);
   }

   PGX_LOG(RUNTIME, DEBUG, "printPtr[%p+%d, len=%d]: %s", ptr, offset, len, hexbuf);
}
