#ifndef PGX_LOWER_RUNTIME_ARROWTABLE_H
#define PGX_LOWER_RUNTIME_ARROWTABLE_H

#include <memory>
#include <vector>
#include <string>

namespace pgx_lower::compiler::runtime {

class ArrowTable {
public:
    ArrowTable() = default;
    virtual ~ArrowTable() = default;
    
    // Stub implementation for Arrow table functionality
    size_t numRows() const { return 0; }
    size_t numColumns() const { return 0; }
};

} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_ARROWTABLE_H