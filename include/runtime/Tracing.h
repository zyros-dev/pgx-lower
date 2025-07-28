#ifndef LINGODB_RUNTIME_TRACING_H
#define LINGODB_RUNTIME_TRACING_H

#include "helpers.h"

namespace pgx_lower::compiler::runtime {
class ExecutionStepTracing {
   public:
   static uint8_t* start(runtime::VarLen32 step);
   static void end(uint8_t* tracing);
};
}; // namespace pgx_lower::compiler::runtime

#endif //LINGODB_RUNTIME_TRACING_H
