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

static int print_call_count = 0;

void runtime::PrintRuntime::printI32(void* label_ptr, int32_t val) {
   // Label is null, just use call count for identification
   PGX_LOG(RUNTIME, DEBUG, "[%d] = %d", ++print_call_count, val);
}

void runtime::PrintRuntime::printNullable(int32_t value, int32_t is_null) {
    PGX_IO(RUNTIME);
   if (is_null) {
      PGX_LOG(RUNTIME, DEBUG, "[%d] = NULL", ++print_call_count);
   } else {
      PGX_LOG(RUNTIME, DEBUG, "[%d] = %d (not null)", ++print_call_count, value);
   }
}
