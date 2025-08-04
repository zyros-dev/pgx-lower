#pragma once

#include "SubOpToControlFlowCommon.h"
#include "SubOpToControlFlowRewriter.h"

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Template pattern classes - implementation now that SubOpRewriter is defined
template <class OpT>
mlir::LogicalResult SubOpConversionPattern<OpT>::matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) {
   std::vector<mlir::Value> newOperands;
   for (auto operand : op->getOperands()) {
      newOperands.push_back(rewriter.getMapped(operand));
   }
   OpAdaptor adaptor(newOperands);
   return matchAndRewrite(mlir::cast<OpT>(op), adaptor, rewriter);
}

template <typename OpType, int numConsumerParams>
mlir::LogicalResult SubOpTupleStreamConsumerConversionPattern<OpType, numConsumerParams>::matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) {
   auto castedOp = mlir::cast<OpType>(op);
   auto stream = castedOp.getStream();
   return rewriter.implementStreamConsumer(stream, [&](SubOpRewriter& rewriter, ColumnMapping& mapping) {
      std::vector<mlir::Value> newOperands;
      for (auto operand : op->getOperands()) {
         newOperands.push_back(rewriter.getMapped(operand));
      }
      OpAdaptor adaptor(newOperands);
      return matchAndRewrite(castedOp, adaptor, rewriter, mapping);
   });
}

// Pattern registration functions
void populateSubOpToControlFlowConversionPatterns(mlir::RewritePatternSet& patterns, 
                                                  mlir::TypeConverter& typeConverter,
                                                  mlir::MLIRContext* context);

void populateTableOperationPatterns(mlir::RewritePatternSet& patterns, 
                                    mlir::TypeConverter& typeConverter,
                                    mlir::MLIRContext* context);

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower