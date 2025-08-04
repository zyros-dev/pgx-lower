#include "execution/mlir_logger.h"
#include <iostream>

// Console logger implementation for standalone usage
void ConsoleLogger::notice(const std::string& message) {
    std::cout << "[NOTICE] " << message << std::endl;
}

void ConsoleLogger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

void ConsoleLogger::debug(const std::string& message) {
    std::cerr << "[DEBUG] " << message << std::endl;
}