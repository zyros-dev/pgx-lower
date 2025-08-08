#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"

namespace pgx {
namespace mlir {
namespace relalg {

// Get or create a column identified by scope and attribute name
std::shared_ptr<Column> ColumnManager::get(const std::string& scope, const std::string& attribute) {
    auto key = std::make_pair(scope, attribute);
    auto it = attributes.find(key);
    if (it != attributes.end()) {
        return it->second;
    }
    
    // Create new column with default type
    auto col = std::make_shared<Column>();
    attributes[key] = col;
    attributesRev[col.get()] = key;
    MLIR_PGX_DEBUG("RelAlg", "Created new column: " + scope + "." + attribute);
    return col;
}

// Get the (scope, name) pair for a column
std::pair<std::string, std::string> ColumnManager::getName(const Column* attr) {
    auto it = attributesRev.find(attr);
    if (it != attributesRev.end()) {
        return it->second;
    }
    return std::make_pair("", "");
}

// TODO: Implement these methods when ColumnDefAttr and ColumnRefAttr are available
/*
// Create column definition attribute
ColumnDefAttr ColumnManager::createDef(const Column* col) {
    // TODO: Implement when ColumnDefAttr is available
    return ColumnDefAttr();
}

ColumnDefAttr ColumnManager::createDef(const std::string& scope, const std::string& name, 
                                       ::mlir::Attribute fromExisting) {
    auto col = get(scope, name);
    return createDef(col.get());
}

// Create column reference attribute  
ColumnRefAttr ColumnManager::createRef(const Column* col) {
    // TODO: Implement when ColumnRefAttr is available
    return ColumnRefAttr();
}

ColumnRefAttr ColumnManager::createRef(const std::string& scope, const std::string& name) {
    auto col = get(scope, name);
    return createRef(col.get());
}
*/

} // namespace relalg
} // namespace mlir
} // namespace pgx