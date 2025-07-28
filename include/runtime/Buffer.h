#ifndef PGX_LOWER_RUNTIME_BUFFER_H
#define PGX_LOWER_RUNTIME_BUFFER_H

#include <memory>
#include <vector>
#include <cstddef>

namespace pgx_lower::compiler::runtime {

class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t size) : data_(size) {}
    virtual ~Buffer() = default;
    
    // Stub implementation for buffer functionality
    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    void resize(size_t newSize) { data_.resize(newSize); }

private:
    std::vector<char> data_;
};

} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_BUFFER_H