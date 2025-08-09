#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/IR/Value.h"
#include <unordered_map>
#include <memory>

namespace pgx {
namespace mlir {
namespace relalg {

// TranslatorContext manages value mappings and scoping during RelAlg->DB translation
class TranslatorContext {
private:
    // Scoped symbol table for column attribute resolution
    llvm::ScopedHashTable<const Column*, ::mlir::Value> symbolTable;
    
    // Shared ColumnManager for ensuring column identity across translators
    std::shared_ptr<ColumnManager> columnManager;

public:
    using AttributeResolverScope = llvm::ScopedHashTableScope<const Column*, ::mlir::Value>;
    
    // Constructor initializes the shared column manager
    TranslatorContext() : columnManager(std::make_shared<ColumnManager>()) {}
    
    // Get the shared column manager
    std::shared_ptr<ColumnManager> getColumnManager() const {
        return columnManager;
    }
    
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
    
    // Query result management
    ::mlir::Value queryResult;
    
    void setQueryResult(::mlir::Value result) {
        queryResult = result;
    }
    
    ::mlir::Value getQueryResult() const {
        return queryResult;
    }
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H