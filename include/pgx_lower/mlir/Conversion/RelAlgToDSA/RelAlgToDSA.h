#ifndef PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H
#define PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <memory>

namespace mlir {
namespace pgx_conversion {

/// Create a pass that lowers RelAlg dialect operations to DSA dialect operations.
/// This pass converts:
/// - relalg.basetable -> dsa.scan_source  
/// - relalg.materialize -> DSA result builder pattern (create_ds, ds_append, next_row, finalize)
/// - relalg.return -> proper terminator handling
std::unique_ptr<Pass> createRelAlgToDSAPass();

/// Register RelAlg to DSA conversion passes
void registerRelAlgToDSAConversionPasses();

//===----------------------------------------------------------------------===//
// Lowering Pattern Classes (exposed for testing)
//===----------------------------------------------------------------------===//

/// Pattern to convert relalg.basetable to dsa.scan_source
struct BaseTableToScanSourcePattern : public OpRewritePattern<::pgx::mlir::relalg::BaseTableOp> {
    BaseTableToScanSourcePattern(MLIRContext *context) : OpRewritePattern<::pgx::mlir::relalg::BaseTableOp>(context) {}
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::BaseTableOp op, PatternRewriter &rewriter) const override;
};

/// Pattern to convert relalg.materialize to DSA result builder pattern
struct MaterializeToResultBuilderPattern : public OpRewritePattern<::pgx::mlir::relalg::MaterializeOp> {
    MaterializeToResultBuilderPattern(MLIRContext *context) : OpRewritePattern<::pgx::mlir::relalg::MaterializeOp>(context) {}
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, PatternRewriter &rewriter) const override;
};

/// Pattern to convert relalg.return to proper terminator
struct ReturnOpLoweringPattern : public OpRewritePattern<::pgx::mlir::relalg::ReturnOp> {
    ReturnOpLoweringPattern(MLIRContext *context) : OpRewritePattern<::pgx::mlir::relalg::ReturnOp>(context) {}
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::ReturnOp op, PatternRewriter &rewriter) const override;
};

} // namespace pgx_conversion
} // namespace mlir

#endif // PGX_MLIR_CONVERSION_RELALGTODSA_RELALGTODSA_H