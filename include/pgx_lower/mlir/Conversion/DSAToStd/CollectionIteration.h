#ifndef PGX_LOWER_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#define PGX_LOWER_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace pgx::mlir::dsa {
class CollectionIterationImpl {
   public:
   virtual std::vector<::mlir::Value> implementLoop(::mlir::Location loc, ::mlir::ValueRange iterArgs, ::mlir::Value flag, ::mlir::TypeConverter& typeConverter, ::mlir::ConversionPatternRewriter& builder, ::mlir::ModuleOp parentModule, std::function<std::vector<::mlir::Value>(std::function<::mlir::Value(::mlir::OpBuilder&)>,::mlir::ValueRange, ::mlir::OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl() {
   }
   static std::unique_ptr<pgx::mlir::dsa::CollectionIterationImpl> getImpl(::mlir::Type collectionType, ::mlir::Value loweredCollection);
};

} // namespace pgx::mlir::dsa
#endif // PGX_LOWER_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H