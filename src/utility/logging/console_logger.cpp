#include "execution/mlir_logger.h"
#include "execution/logging.h"
#include <iostream>

// Console logger implementation for standalone usage
void ConsoleLogger::notice(const std::string& message) {
    PGX_NOTICE(message);
}

void ConsoleLogger::error(const std::string& message) {
    PGX_ERROR(message);
}

void ConsoleLogger::debug(const std::string& message) {
    PGX_DEBUG(message);
}