#ifndef MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#define MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "lingodb/mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/IR/Value.h"
#include "pgx-lower/utility/logging.h"
namespace mlir {
namespace relalg {
class TranslatorContext {
   llvm::ScopedHashTable<const mlir::relalg::Column*, ::mlir::Value> symbolTable;

   public:
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::Column*, ::mlir::Value>;

   ::mlir::Value getValueForAttribute(const mlir::relalg::Column* attribute) const {
       const auto value = symbolTable.lookup(attribute);
      if (!value) {
          std::string typeStr = "unknown";
          if (attribute && attribute->type) {
              llvm::raw_string_ostream os(typeStr);
              attribute->type.print(os);
          }
          PGX_ERROR("Column lookup failed - type: %s, ptr: %p. Check that child operations properly populate column schemas.",
                   typeStr.c_str(), attribute);
          assert(false);
      }
      return value;
   }
   ::mlir::Value getUnsafeValueForAttribute(const mlir::relalg::Column* attribute) const {
      return symbolTable.lookup(attribute);
   }
   void setValueForAttribute(AttributeResolverScope& scope, const mlir::relalg::Column* iu, ::mlir::Value v) {
      symbolTable.insertIntoScope(&scope, iu, v);
   }
   AttributeResolverScope createScope() {
      return AttributeResolverScope(symbolTable);
   }
   std::unordered_map<size_t, ::mlir::Value> builders;

   std::unordered_map<::mlir::Operation*, std::pair<::mlir::Value, std::vector<const mlir::relalg::Column*>>> materializedTmp;
};
}
}

#endif
