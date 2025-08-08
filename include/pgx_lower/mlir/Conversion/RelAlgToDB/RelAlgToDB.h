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