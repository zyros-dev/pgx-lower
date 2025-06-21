#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include "mlir_logger.h"
#include <cstdint>

namespace mlir_runner {

bool run_mlir_core(int64_t intValue, MLIRLogger& logger);

#ifndef POSTGRESQL_EXTENSION
bool run_mlir_test(int64_t intValue);
#endif

}

#endif // MLIR_RUNNER_H