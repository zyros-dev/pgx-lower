#ifndef PGX_LOWER_RUNTIME_EXECUTIONCONTEXT_H
#define PGX_LOWER_RUNTIME_EXECUTIONCONTEXT_H

namespace pgx_lower::compiler::runtime {

class ExecutionContext {
public:
    ExecutionContext() = default;
    virtual ~ExecutionContext() = default;
    
    // Stub implementation
    void reset() {}
    bool isInitialized() { return true; }
};

} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_EXECUTIONCONTEXT_H