#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/IR/Value.h"
#include <unordered_map>

namespace pgx {
namespace mlir {
namespace relalg {

// TranslatorContext manages value mappings and scoping during RelAlg->DB translation
class TranslatorContext {
private:
    // Scoped symbol table for column attribute resolution
    llvm::ScopedHashTable<const Column*, ::mlir::Value> symbolTable;

public:
    using AttributeResolverScope = llvm::ScopedHashTableScope<const Column*, ::mlir::Value>;
    
    // Column attribute resolution functions
    ::mlir::Value getValueForAttribute(const Column* attribute) const {
        assert(symbolTable.count(attribute) && "Column not found in symbol table");
        return symbolTable.lookup(attribute);
    }
    
    ::mlir::Value getUnsafeValueForAttribute(const Column* attribute) const {
        return symbolTable.lookup(attribute);
    }
    
    void setValueForAttribute(AttributeResolverScope& scope, const Column* col, ::mlir::Value v) {
        symbolTable.insertIntoScope(&scope, col, v);
    }
    
    AttributeResolverScope createScope() {
        return AttributeResolverScope(symbolTable);
    }
    
    // Storage for DSA builders and materialized temporary values
    std::unordered_map<size_t, ::mlir::Value> builders;
    std::unordered_map<::mlir::Operation*, std::pair<::mlir::Value, std::vector<const Column*>>> materializedTmp;
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H