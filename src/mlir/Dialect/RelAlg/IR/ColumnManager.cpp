#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"

using namespace pgx::mlir::relalg;

ColumnRefAttr ColumnManager::createRef(const std::string& name) {
    // Ensure column exists
    auto it = columns.find(name);
    if (it == columns.end()) {
        // Create placeholder column
        auto column = std::make_shared<Column>(name, ::mlir::NoneType::get(context));
        columns[name] = column;
        PGX_DEBUG("Created placeholder column: " + name);
    }
    
    return ColumnRefAttr::get(context, name);
}