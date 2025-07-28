#ifndef PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#define PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace pgx_lower::compiler::dialect::relalg {
std::unique_ptr<mlir::Pass> createLowerToSubOpPass();
void registerRelAlgToSubOpConversionPasses();
void createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm);

// Additional pass creator that returns OperationPass<ModuleOp>
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerRelAlgToSubOpPass();

// Pattern population
void populateRelAlgToSubOpConversionPatterns(mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter);

} // end namespace pgx_lower::compiler::dialect::relalg
#endif //PGX_LOWER_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
