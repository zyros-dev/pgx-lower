#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include <string>
#include <unordered_map>

namespace pgx::mlir::relalg {

// Forward declarations
class ColumnDefAttr;
class ColumnRefAttr;

struct Column {
    std::string name;
    ::mlir::Type type;
    
    Column(const std::string& name, ::mlir::Type type) : name(name), type(type) {}
};

class ColumnManager {
public:
    ColumnManager() = default;
    
    void setContext(::mlir::MLIRContext* context) { 
        this->context = context; 
    }
    
    ColumnDefAttr createDef(::mlir::SymbolRefAttr name, ::mlir::Attribute fromExisting = nullptr);
    ColumnRefAttr createRef(::mlir::SymbolRefAttr name);
    
private:
    ::mlir::MLIRContext* context = nullptr;
    std::unordered_map<std::string, std::shared_ptr<Column>> columns;
};

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMNMANAGER_H