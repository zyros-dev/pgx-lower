#ifndef PGX_LOWER_RUNTIME_RELATIONHELPER_H
#define PGX_LOWER_RUNTIME_RELATIONHELPER_H

namespace pgx_lower::compiler::runtime {
class RelationHelper {
public:
    RelationHelper() = default;
    virtual ~RelationHelper() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_RELATIONHELPER_H
