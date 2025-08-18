#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dsa {

// Forward declarations to minimize template instantiation
class DSADialect;
class ScanSource;

// Pattern registration functions (implemented in separate compilation units)
void populateScalarToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns);
void populateDSAToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns);
void populateCollectionsToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns);

// Main pass creation
std::unique_ptr<Pass> createLowerToStdPass();

} // namespace dsa
} // namespace mlir