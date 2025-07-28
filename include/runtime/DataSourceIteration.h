#ifndef PGX_LOWER_RUNTIME_DATASOURCEITERATION_H
#define PGX_LOWER_RUNTIME_DATASOURCEITERATION_H

namespace pgx_lower::compiler::runtime {

class DataSourceIteration {
public:
    DataSourceIteration() = default;
    virtual ~DataSourceIteration() = default;
    
    // Stub implementation
    void initialize() {}
    bool hasNext() { return false; }
    void next() {}
};

} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_DATASOURCEITERATION_H