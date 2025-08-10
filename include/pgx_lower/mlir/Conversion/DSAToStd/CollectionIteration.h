#ifndef MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#define MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace pgx::mlir::dsa {
class CollectionIterationImpl {
   public:
   virtual std::vector<Value> implementLoop(mlir::Location loc,mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, mlir::ModuleOp parentModule, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>,ValueRange, OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl() {
   }
   static std::unique_ptr<pgx::mlir::dsa::CollectionIterationImpl> getImpl(mlir::Type collectionType,mlir::Value loweredCollection);
};

} // namespace pgx::mlir::dsa
#endif // MLIR_CONVERSION_DSATOSTD_COLLECTIONITERATION_H
