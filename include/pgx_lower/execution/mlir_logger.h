#ifndef MLIR_LOGGER_H
#define MLIR_LOGGER_H

#include <string>

class MLIRLogger {
   public:
    virtual ~MLIRLogger() = default;
    virtual void notice(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void debug(const std::string& message) = 0;
};

class ConsoleLogger final : public MLIRLogger {
   public:
    void notice(const std::string& message) override;
    void error(const std::string& message) override;
    void debug(const std::string& message) override;
};

class PostgreSQLLogger final : public MLIRLogger {
   public:
    void notice(const std::string& message) override;
    void error(const std::string& message) override;
    void debug(const std::string& message) override;
};

#endif // MLIR_LOGGER_H