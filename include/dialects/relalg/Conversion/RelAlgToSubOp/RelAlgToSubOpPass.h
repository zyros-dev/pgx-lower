#ifndef PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#define PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace pgx_lower::compiler::dialect::relalg {
std::unique_ptr<mlir::Pass> createLowerToSubOpPass();
void registerRelAlgToSubOpConversionPasses();
void createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm);

// Pattern population
void populateRelAlgToSubOpConversionPatterns(mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter);

} // end namespace pgx_lower::compiler::dialect::relalg
#endif //PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
