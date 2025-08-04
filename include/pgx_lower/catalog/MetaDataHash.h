#ifndef PGX_LOWER_CATALOG_METADATA_HASH_H
#define PGX_LOWER_CATALOG_METADATA_HASH_H

#include <memory>

// Forward declaration
namespace pgx_lower::catalog {
class TableMetaDataProvider;
}

// This must be before including Hashing.h to be found by ADL
namespace std {
template<>
struct hash<std::shared_ptr<pgx_lower::catalog::TableMetaDataProvider>> {
    size_t operator()(const std::shared_ptr<pgx_lower::catalog::TableMetaDataProvider>& ptr) const {
        return std::hash<void*>()(ptr.get());
    }
};
} // namespace std

#include "llvm/ADT/Hashing.h"

// Provide hash_value for shared_ptr<TableMetaDataProvider>
namespace llvm {
inline hash_code hash_value(const std::shared_ptr<pgx_lower::catalog::TableMetaDataProvider>& ptr) {
    // Hash based on the pointer value
    return hash_combine(ptr.get());
}
} // namespace llvm

#endif // PGX_LOWER_CATALOG_METADATA_HASH_H