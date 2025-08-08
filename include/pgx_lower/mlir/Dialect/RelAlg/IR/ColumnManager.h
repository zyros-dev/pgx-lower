#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace pgx {
namespace mlir {
namespace relalg {

// Forward declarations
class ColumnRefAttr;
class ColumnDefAttr;

// ColumnManager manages the creation and tracking of Column objects
// It ensures columns are uniquely identified by scope and name
class ColumnManager {
private:
    ::mlir::MLIRContext* context = nullptr;
    
    // Hash function for pair<string, string>
    struct HashPair {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ hash2;
        }
    };
    
    // Maps from (scope, name) to Column
    std::unordered_map<std::pair<std::string, std::string>, std::shared_ptr<Column>, HashPair> attributes;
    
    // Reverse map from Column to (scope, name)
    std::unordered_map<const Column*, std::pair<std::string, std::string>> attributesRev;
    
    // Scope uniquifier for generating unique scope names
    std::unordered_map<std::string, size_t> scopeUnifier;

public:
    ColumnManager() = default;
    
    void setContext(::mlir::MLIRContext* context) { 
        this->context = context; 
    }
    
    // Get or create a column identified by scope and attribute name
    std::shared_ptr<Column> get(const std::string& scope, const std::string& attribute);
    
    // TODO: Implement these methods when ColumnDefAttr and ColumnRefAttr are available
    /*
    // Create column definition attribute
    ColumnDefAttr createDef(const Column* col);
    ColumnDefAttr createDef(const std::string& scope, const std::string& name, 
                           ::mlir::Attribute fromExisting = {});
    
    // Create column reference attribute
    ColumnRefAttr createRef(const Column* col);
    ColumnRefAttr createRef(const std::string& scope, const std::string& name);
    */
    
    // Get the (scope, name) pair for a column
    std::pair<std::string, std::string> getName(const Column* attr);
    
    // Generate a unique scope name based on a base name
    std::string getUniqueScope(const std::string& base) {
        if (scopeUnifier.count(base)) {
            scopeUnifier[base] += 1;
            return base + std::to_string(scopeUnifier[base]);
        } else {
            scopeUnifier[base] = 0;
            return base;
        }
    }
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H