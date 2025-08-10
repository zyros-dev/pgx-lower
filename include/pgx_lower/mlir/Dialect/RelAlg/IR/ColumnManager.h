#ifndef MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H
#define MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H

#include "mlir/Dialect/RelAlg/IR/Column.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"

namespace pgx::mlir::relalg {
class ColumnManager {
   public:
   void setContext(::mlir::MLIRContext* context);
   std::shared_ptr<Column> get(::llvm::StringRef scope, ::llvm::StringRef attribute);
   ColumnDefAttr createDef(::mlir::SymbolRefAttr name, ::mlir::Attribute fromExisting = ::mlir::Attribute());
   ColumnDefAttr createDef(::llvm::StringRef scope, ::llvm::StringRef name, ::mlir::Attribute fromExisting = ::mlir::Attribute());
   ColumnDefAttr createDef(const Column* attr);

   ColumnRefAttr createRef(::mlir::SymbolRefAttr name);
   ColumnRefAttr createRef(const Column* attr);

   ColumnRefAttr createRef(::llvm::StringRef scope, ::llvm::StringRef name);
   std::pair<std::string, std::string> getName(const Column* attr);

   std::string getUniqueScope(::llvm::StringRef base) {
      if (scopeUnifier.count(std::string(base))) {
         scopeUnifier[std::string(base)] += 1;
         return std::string(base) + std::to_string(scopeUnifier[std::string(base)]);
      } else {
         scopeUnifier[std::string(base)] = 0;
         return std::string(base);
      }
   }

   private:
   ::mlir::MLIRContext* context;
   struct HashPair {
      template <class T1, class T2>
      size_t operator()(const std::pair<T1, T2>& p) const {
         auto hash1 = std::hash<T1>{}(p.first);
         auto hash2 = std::hash<T2>{}(p.second);
         return hash1 ^ hash2;
      }
   };
   std::unordered_map<std::pair<std::string, std::string>, std::shared_ptr<Column>, HashPair> attributes;
   std::unordered_map<const Column*, std::pair<std::string, std::string>> attributesRev;

   std::unordered_map<std::string, size_t> scopeUnifier;
};
} // namespace pgx::mlir::relalg

#endif // MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H
