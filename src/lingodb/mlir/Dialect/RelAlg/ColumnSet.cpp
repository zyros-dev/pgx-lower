#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"

namespace mlir::relalg {

::mlir::ArrayAttr ColumnSet::asRefArrayAttr(::mlir::MLIRContext* context) {
    auto& columnManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
    
    std::vector<::mlir::Attribute> refAttrs;
    for (const auto* col : columns) {
        refAttrs.push_back(columnManager.createRef(col));
    }
    return ::mlir::ArrayAttr::get(context, refAttrs);
}

ColumnSet ColumnSet::fromArrayAttr(::mlir::ArrayAttr arrayAttr) {
    ColumnSet res;
    for (const auto attr : arrayAttr) {
        if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>()) {
            res.insert(&attrRef.getColumn());
        } else if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>()) {
            res.insert(&attrDef.getColumn());
        }
    }
    return res;
}

} // namespace mlir::relalg