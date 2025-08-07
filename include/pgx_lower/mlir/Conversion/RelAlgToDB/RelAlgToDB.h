#ifndef MLIR_CONVERSION_RELALGTODB_RELALGTODB_H
#define MLIR_CONVERSION_RELALGTODB_RELALGTODB_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <memory>

namespace mlir {

class Pass;

namespace pgx_conversion {

//===----------------------------------------------------------------------===//
// RelAlg to DB Lowering Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert RelAlg BaseTableOp to DB GetExternalOp
struct BaseTableToExternalSourcePattern : public OpConversionPattern<::pgx::mlir::relalg::BaseTableOp> {
    using OpConversionPattern<::pgx::mlir::relalg::BaseTableOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::BaseTableOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert RelAlg GetColumnOp to DB GetFieldOp
struct GetColumnToGetFieldPattern : public OpConversionPattern<::pgx::mlir::relalg::GetColumnOp> {
    using OpConversionPattern<::pgx::mlir::relalg::GetColumnOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::GetColumnOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert RelAlg MaterializeOp to DB result streaming operations
/// 
/// CRITICAL: This pattern is DISABLED in Phase 3a per LingoDB research findings.
/// MaterializeOp creates DSA operations (dsa.create_ds, dsa.ds_append, etc.), not DB operations.
/// MaterializeOp belongs in Phase 3b (DB→DSA), not Phase 3a (RelAlg→DB).
/// In Phase 3a, MaterializeOp is marked as LEGAL and passes through unchanged.
struct MaterializeToStreamResultsPattern : public OpConversionPattern<::pgx::mlir::relalg::MaterializeOp> {
    using OpConversionPattern<::pgx::mlir::relalg::MaterializeOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert RelAlg ReturnOp to func.return
struct ReturnOpToFuncReturnPattern : public OpConversionPattern<::pgx::mlir::relalg::ReturnOp> {
    using OpConversionPattern<::pgx::mlir::relalg::ReturnOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::ReturnOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

/// Create the RelAlg to DB conversion pass
std::unique_ptr<Pass> createRelAlgToDBPass();

/// Register RelAlg to DB conversion passes
void registerRelAlgToDBConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODB_H