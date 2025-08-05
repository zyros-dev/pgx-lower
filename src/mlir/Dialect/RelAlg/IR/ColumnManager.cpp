#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"

using namespace pgx::mlir::relalg;

ColumnDefAttr ColumnManager::createDef(::mlir::SymbolRefAttr name, ::mlir::Attribute fromExisting) {
    std::string nameStr = name.str();
    
    // Create or find column
    auto it = columns.find(nameStr);
    if (it == columns.end()) {
        // Create new column with placeholder type
        auto column = std::make_shared<Column>(nameStr, ::mlir::NoneType::get(context));
        columns[nameStr] = column;
    }
    
    return ColumnDefAttr::get(context, name, ::mlir::NoneType::get(context), fromExisting);
}

ColumnRefAttr ColumnManager::createRef(::mlir::SymbolRefAttr name) {
    std::string nameStr = name.str();
    
    // Ensure column exists
    auto it = columns.find(nameStr);
    if (it == columns.end()) {
        // Create placeholder column
        auto column = std::make_shared<Column>(nameStr, ::mlir::NoneType::get(context));
        columns[nameStr] = column;
        PGX_DEBUG("Created placeholder column: " + nameStr);
    }
    
    return ColumnRefAttr::get(context, name);
}