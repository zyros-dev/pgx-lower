#include "execution/mlir_logger.h"
#include "execution/logging.h"

// Prevent libintl.h conflicts with PostgreSQL macros
// This is a bit strange to me - so LLVM drags in some macros from libintl.h
// and those conflict with things inside libintl.h. So this should resolve
// those problems?
#define ENABLE_NLS 0

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

// PostgreSQL logger implementation for the extension
void PostgreSQLLogger::notice(const std::string& message) {
    PGX_NOTICE(message);
}

void PostgreSQLLogger::error(const std::string& message) {
    PGX_ERROR(message);
}

void PostgreSQLLogger::debug(const std::string& message) {
    PGX_DEBUG(message);
}