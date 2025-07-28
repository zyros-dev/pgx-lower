#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "mlir/IR/BuiltinTypes.h"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.h.inc"

namespace pgx_lower::compiler::dialect::tuples {

ColumnDefAttr ColumnManager::createDef(mlir::SymbolRefAttr name, mlir::Attribute fromExisting) {
    assert(name.getNestedReferences().size() == 1);
    auto column = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
    // Use a default type for now - TODO: properly track column types
    mlir::Type columnType = mlir::IntegerType::get(context, 32);
    return ColumnDefAttr::get(context, name, columnType, fromExisting);
}

ColumnDefAttr ColumnManager::createDef(llvm::StringRef scope, llvm::StringRef name, mlir::Attribute fromExisting) {
    auto column = get(scope, name);
    std::vector<mlir::FlatSymbolRefAttr> nested;
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, name));
    // Use a default type for now - TODO: properly track column types
    mlir::Type columnType = mlir::IntegerType::get(context, 32);
    return ColumnDefAttr::get(context, mlir::SymbolRefAttr::get(context, scope, nested), columnType, fromExisting);
}

ColumnDefAttr ColumnManager::createDef(const Column* attr, mlir::Attribute fromExisting) {
    auto [scope, name] = attributesRev[attr];
    return createDef(scope, name, fromExisting);
}

ColumnRefAttr ColumnManager::createRef(mlir::SymbolRefAttr name) {
    assert(name.getNestedReferences().size() == 1);
    auto column = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
    // Use a default type for now - TODO: properly track column types
    mlir::Type columnType = mlir::IntegerType::get(context, 32);
    return ColumnRefAttr::get(context, name, columnType);
}

ColumnRefAttr ColumnManager::createRef(const Column* attr) {
    auto [scope, name] = attributesRev[attr];
    return createRef(scope, name);
}

ColumnRefAttr ColumnManager::createRef(llvm::StringRef scope, llvm::StringRef name) {
    auto column = get(scope, name);
    std::vector<mlir::FlatSymbolRefAttr> nested;
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, name));
    // Use a default type for now - TODO: properly track column types
    mlir::Type columnType = mlir::IntegerType::get(context, 32);
    return ColumnRefAttr::get(context, mlir::SymbolRefAttr::get(context, scope, nested), columnType);
}

} // namespace pgx_lower::compiler::dialect::tuples