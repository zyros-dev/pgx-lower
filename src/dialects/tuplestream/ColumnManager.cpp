#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "mlir/IR/BuiltinTypes.h"

// Attributes are already included via ColumnManager.h

namespace pgx_lower::compiler::dialect::tuples {

ColumnDefAttr ColumnManager::createDef(mlir::SymbolRefAttr name, mlir::Attribute fromExisting) {
    assert(name.getNestedReferences().size() == 1);
    auto column = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
    return ColumnDefAttr::get(context, name, column, fromExisting);
}

ColumnDefAttr ColumnManager::createDef(llvm::StringRef scope, llvm::StringRef name, mlir::Attribute fromExisting) {
    auto column = get(scope, name);
    std::vector<mlir::FlatSymbolRefAttr> nested;
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, name));
    return ColumnDefAttr::get(context, mlir::SymbolRefAttr::get(context, scope, nested), column, fromExisting);
}

ColumnDefAttr ColumnManager::createDef(const Column* attr, mlir::Attribute fromExisting) {
    auto [scope, name] = attributesRev[attr];
    return createDef(scope, name, fromExisting);
}

ColumnRefAttr ColumnManager::createRef(mlir::SymbolRefAttr name) {
    assert(name.getNestedReferences().size() == 1);
    auto column = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
    return ColumnRefAttr::get(context, name, column);
}

ColumnRefAttr ColumnManager::createRef(const Column* attr) {
    auto [scope, name] = attributesRev[attr];
    return createRef(scope, name);
}

ColumnRefAttr ColumnManager::createRef(llvm::StringRef scope, llvm::StringRef name) {
    auto column = get(scope, name);
    std::vector<mlir::FlatSymbolRefAttr> nested;
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, name));
    return ColumnRefAttr::get(context, mlir::SymbolRefAttr::get(context, scope, nested), column);
}

} // namespace pgx_lower::compiler::dialect::tuples