#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include "mlir_logger.h"
#include <cstdint>
#include <functional>

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

bool run_mlir_core(int64_t intValue, MLIRLogger& logger);
bool run_mlir_with_external_func(int64_t intValue, const ExternalFunction& externalFunc, MLIRLogger& logger);

#ifndef POSTGRESQL_EXTENSION
bool run_mlir_test(int64_t intValue);
bool run_external_func_test(const ExternalFunction& externalFunc);
#endif

}

#endif // MLIR_RUNNER_H