#ifndef PGX_LOWER_RUNTIME_QUERYEXECUTOR_H
#define PGX_LOWER_RUNTIME_QUERYEXECUTOR_H

#include <memory>
#include <functional>

namespace pgx_lower::compiler::runtime {

class QueryExecutor {
public:
    QueryExecutor() = default;
    virtual ~QueryExecutor() = default;
    
    // Stub implementation for query execution
    template<typename T>
    void execute(T&& function) {
        function();
    }
};

} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_QUERYEXECUTOR_H