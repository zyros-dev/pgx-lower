#pragma once

#include <cstdint>
#include <utility>
#include "mlir/IR/Types.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

// Forward declarations
namespace mlir {
    class MLIRContext;
}

namespace pgx_lower::frontend::sql {

class PostgreSQLTypeMapper {
public:
    explicit PostgreSQLTypeMapper(::mlir::MLIRContext& context);

    // Extract character length from typmod for CHAR/VARCHAR
    int32_t extractCharLength(int32_t typmod);

    // Extract precision and scale from NUMERIC typmod
    std::pair<int32_t, int32_t> extractNumericInfo(int32_t typmod);

    // Extract timestamp precision from typmod
    mlir::db::TimeUnitAttr extractTimestampPrecision(int32_t typmod);

    // Map PostgreSQL type OID to MLIR type
    ::mlir::Type mapPostgreSQLType(unsigned int typeOid, int32_t typmod);

private:
    ::mlir::MLIRContext& context_;
};

} // namespace pgx_lower::frontend::sql