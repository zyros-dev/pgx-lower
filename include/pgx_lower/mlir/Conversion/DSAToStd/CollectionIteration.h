#ifndef PGX_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#define PGX_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::dsa {
class CollectionIterationImpl {
   public:
   virtual std::vector<::mlir::Value> implementLoop(::mlir::Location loc,::mlir::ValueRange iterArgs, ::mlir::Value flag, const ::mlir::TypeConverter& typeConverter, ::mlir::ConversionPatternRewriter& builder, ::mlir::ModuleOp parentModule, std::function<std::vector<::mlir::Value>(std::function<::mlir::Value(::mlir::OpBuilder&)>,::mlir::ValueRange, ::mlir::OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl() {
   }
   static std::unique_ptr<CollectionIterationImpl> getImpl(::mlir::Type collectionType,::mlir::Value loweredCollection);
};

} // namespace mlir::dsa
#endif // PGX_MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
