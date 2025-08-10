#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"

using namespace mlir;
using namespace llvm;

namespace pgx::mlir::relalg {
void ColumnManager::setContext(::mlir::MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<Column> ColumnManager::get(::llvm::StringRef scope, ::llvm::StringRef attribute) {
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(pair)) {
      auto attr = std::make_shared<Column>();
      attributes[pair] = attr;
      attributesRev[attr.get()] = pair;
   }
   return attributes[pair];
}
ColumnDefAttr ColumnManager::createDef(::mlir::SymbolRefAttr name, ::mlir::Attribute fromExisting) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return pgx::mlir::relalg::ColumnDefAttr::get(context, name, attribute, fromExisting);
}
ColumnDefAttr ColumnManager::createDef(::llvm::StringRef scope, ::llvm::StringRef name, ::mlir::Attribute fromExisting) {
   auto attribute = get(scope, name);
   std::vector<::mlir::FlatSymbolRefAttr> nested;
   nested.push_back(::mlir::FlatSymbolRefAttr::get(context, name));
   return pgx::mlir::relalg::ColumnDefAttr::get(context, ::mlir::SymbolRefAttr::get(context, scope, nested), attribute, fromExisting);
}
ColumnRefAttr ColumnManager::createRef(::mlir::SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return relalg::ColumnRefAttr::get(context, name, attribute);
}
ColumnRefAttr ColumnManager::createRef(::llvm::StringRef scope, ::llvm::StringRef name) {
   auto attribute = get(scope, name);
   std::vector<::mlir::FlatSymbolRefAttr> nested;
   nested.push_back(::mlir::FlatSymbolRefAttr::get(context, name));
   return relalg::ColumnRefAttr::get(context, ::mlir::SymbolRefAttr::get(context, scope, nested), attribute);
}
ColumnRefAttr ColumnManager::createRef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   return createRef(scope, name);
}
ColumnDefAttr ColumnManager::createDef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   return createDef(scope, name);
}

std::pair<std::string, std::string> ColumnManager::getName(const Column* attr) {
   return attributesRev.at(attr);
}
} // namespace pgx::mlir::relalg
