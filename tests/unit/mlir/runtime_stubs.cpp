// Runtime stubs for unit testing
// These are minimal implementations to satisfy linker dependencies
// without pulling in the full PostgreSQL runtime

#include <string>
#include <memory>
#include "runtime/metadata.h"

// Forward declarations needed
namespace mlir {
class TypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace runtime {

// Implement the methods declared in metadata.h
std::shared_ptr<TableMetaData> TableMetaData::deserialize(std::string str) {
    // Minimal stub for unit testing
    auto result = std::make_shared<TableMetaData>();
    result->present = true;
    return result;
}

std::string TableMetaData::serialize(bool serializeSample) const {
    // Minimal stub for unit testing
    return "test_metadata";
}

bool TableMetaData::isPresent() const {
    // Minimal stub for unit testing
    return present;
}

} // namespace runtime

namespace mlir {
namespace util {

void populateUtilTypeConversionPatterns(TypeConverter& converter, RewritePatternSet& patterns) {
    // Minimal stub for unit testing
    // The real implementation would populate conversion patterns
}

} // namespace util
} // namespace mlir