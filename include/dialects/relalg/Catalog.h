#ifndef PGX_LOWER_CATALOG_H
#define PGX_LOWER_CATALOG_H

namespace catalog {
// Stub for TableMetaDataProvider - TODO Phase 5: implement for PostgreSQL
class TableMetaDataProvider {
public:
    TableMetaDataProvider() = default;
    virtual ~TableMetaDataProvider() = default;
};
} // namespace catalog

#endif // PGX_LOWER_CATALOG_H