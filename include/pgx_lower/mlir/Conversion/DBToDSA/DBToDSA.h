#ifndef MLIR_CONVERSION_DBTODSA_DBTODSA_H
#define MLIR_CONVERSION_DBTODSA_DBTODSA_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include <memory>

namespace mlir {

class Pass;

namespace pgx_conversion {

//===----------------------------------------------------------------------===//
// DB to DSA Lowering Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert DB GetExternalOp to DSA ScanSourceOp
struct GetExternalToScanSourcePattern : public OpConversionPattern<::pgx::db::GetExternalOp> {
    using OpConversionPattern<::pgx::db::GetExternalOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::db::GetExternalOp op,
                                  PatternRewriter &rewriter) const override;
};

/// Pattern to convert DB GetFieldOp to DSA AtOp
struct GetFieldToAtPattern : public OpConversionPattern<::pgx::db::GetFieldOp> {
    using OpConversionPattern<::pgx::db::GetFieldOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::db::GetFieldOp op,
                                  PatternRewriter &rewriter) const override;
};

/// Pattern to convert DB StreamResultsOp to DSA finalization operations
struct StreamResultsToFinalizePattern : public OpConversionPattern<::pgx::db::StreamResultsOp> {
    using OpConversionPattern<::pgx::db::StreamResultsOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::db::StreamResultsOp op,
                                  PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

/// Create the DB to DSA conversion pass
std::unique_ptr<Pass> createDBToDSAPass();

/// Register DB to DSA conversion passes
void registerDBToDSAConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // MLIR_CONVERSION_DBTODSA_DBTODSA_H