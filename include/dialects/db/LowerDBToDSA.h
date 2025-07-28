#ifndef PGX_LOWER_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H
#define PGX_LOWER_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace pgx_lower::compiler::dialect {
namespace db {

std::unique_ptr<mlir::Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

} // end namespace db
} // end namespace pgx_lower::compiler::dialect

#endif //PGX_LOWER_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H