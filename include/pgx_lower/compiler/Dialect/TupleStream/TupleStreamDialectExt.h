#ifndef TUPLESTREAM_DIALECT_EXT_H
#define TUPLESTREAM_DIALECT_EXT_H

#include "ColumnManager.h"

// This header extends the generated TupleStreamDialect class with additional members
namespace pgx_lower::compiler::dialect::tuples {

// Add the columnManager member variable to TupleStreamDialect
// This is a workaround since we can't modify the generated class directly
class TupleStreamDialectWithColumnManager : public TupleStreamDialect {
public:
    using TupleStreamDialect::TupleStreamDialect;
    
    ColumnManager& getColumnManager() {
        return columnManager;
    }
    
private:
    ColumnManager columnManager;
};

} // namespace pgx_lower::compiler::dialect::tuples

#endif // TUPLESTREAM_DIALECT_EXT_H