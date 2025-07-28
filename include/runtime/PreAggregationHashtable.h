#ifndef PGX_LOWER_RUNTIME_PREAGGREGATIONHASHTABLE_H
#define PGX_LOWER_RUNTIME_PREAGGREGATIONHASHTABLE_H

namespace pgx_lower::compiler::runtime {
class PreAggregationHashtable {
public:
    PreAggregationHashtable() = default;
    virtual ~PreAggregationHashtable() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_PREAGGREGATIONHASHTABLE_H
