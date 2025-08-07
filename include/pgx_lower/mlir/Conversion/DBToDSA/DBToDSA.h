#ifndef MLIR_CONVERSION_DBTODSA_DBTODSA_H
#define MLIR_CONVERSION_DBTODSA_DBTODSA_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <memory>

namespace mlir {

class Pass;

namespace pgx_conversion {

//===----------------------------------------------------------------------===//
// DB to DSA Lowering Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert DB GetExternalOp to DSA ScanSourceOp
struct GetExternalToScanSourcePattern : public OpConversionPattern<::pgx::db::GetExternalOp> {
    GetExternalToScanSourcePattern(mlir::TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<::pgx::db::GetExternalOp>(typeConverter, context) {}
    
    LogicalResult matchAndRewrite(::pgx::db::GetExternalOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert DB GetFieldOp to DSA AtOp
struct GetFieldToAtPattern : public OpConversionPattern<::pgx::db::GetFieldOp> {
    using OpConversionPattern<::pgx::db::GetFieldOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(::pgx::db::GetFieldOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert DB StreamResultsOp to DSA finalization operations
struct StreamResultsToFinalizePattern : public OpConversionPattern<::pgx::db::StreamResultsOp> {
    StreamResultsToFinalizePattern(mlir::TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<::pgx::db::StreamResultsOp>(typeConverter, context) {}
    
    LogicalResult matchAndRewrite(::pgx::db::StreamResultsOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert RelAlg MaterializeOp to DSA operations
struct MaterializeToDSAPattern : public OpConversionPattern<::pgx::mlir::relalg::MaterializeOp> {
    MaterializeToDSAPattern(mlir::TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<::pgx::mlir::relalg::MaterializeOp>(typeConverter, context) {}
    
    LogicalResult matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// DB to DSA Type Converter
//===----------------------------------------------------------------------===//

/// Type converter for DB to DSA conversion
class DBToDSATypeConverter : public TypeConverter {
public:
    DBToDSATypeConverter();
    
    /// Get Arrow schema description for a type (for DSA CreateDS)
    std::string getArrowDescription(Type type) const;
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