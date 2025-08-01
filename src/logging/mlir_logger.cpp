#include "core/mlir_logger.h"
#include <iostream>

void ConsoleLogger::notice(const std::string& message) {
    std::cout << "[NOTICE] " << message << std::endl;
}

void ConsoleLogger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

void ConsoleLogger::debug(const std::string& message) {
    std::cout << "[DEBUG] " << message << std::endl;
}

// Only compile when PostgresSQL is available
#ifdef HAVE_POSTGRESQL
extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

void PostgreSQLLogger::notice(const std::string& message) {
    elog(NOTICE, "%s", message.c_str());
}

void PostgreSQLLogger::error(const std::string& message) {
    elog(ERROR, "%s", message.c_str());
}

void PostgreSQLLogger::debug(const std::string& message) {
    elog(LOG, "%s", message.c_str());
}
#else
// Fallback implementation for unit tests
void PostgreSQLLogger::notice(const std::string& message) {
    std::cout << "[PG_NOTICE] " << message << std::endl;
}

void PostgreSQLLogger::error(const std::string& message) {
    std::cerr << "[PG_ERROR] " << message << std::endl;
}

void PostgreSQLLogger::debug(const std::string& message) {
    std::cout << "[PG_DEBUG] " << message << std::endl;
}
#endif