#ifndef MLIR_BUILDER_H
#define MLIR_BUILDER_H

#include "mlir_logger.h"
#include <vector>
#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
class Value;
class OpBuilder;
}

namespace mlir_builder {

// Structure to represent a column or computed expression
struct ColumnExpression {
    int columnIndex;  // -1 for computed expressions, >= 0 for regular columns
    std::string operatorName;  // "+", "-", "*", "/", "%" for arithmetic ops
    std::vector<int> operandColumns;  // Column indices for operands
    std::vector<int> operandConstants;  // Constant values for operands
    
    // Constructor for regular column
    explicit ColumnExpression(int col) : columnIndex(col) {}
    
    // Constructor for arithmetic expression
    ColumnExpression(const std::string& op, const std::vector<int>& cols, const std::vector<int>& consts = {})
        : columnIndex(-1), operatorName(op), operandColumns(cols), operandConstants(consts) {}
};

class MLIRBuilder {
public:
    explicit MLIRBuilder(mlir::MLIRContext& context);
    ~MLIRBuilder() = default;

    // PostgreSQL Plan Node to MLIR conversion
    auto buildTableScanModule(const char* tableName, const std::vector<ColumnExpression>& expressions) 
        -> std::unique_ptr<mlir::ModuleOp>;
    
    // PostgreSQL Plan Node to MLIR conversion with WHERE clause support
    auto buildTableScanModuleWithWhere(const char* tableName, const std::vector<ColumnExpression>& expressions,
                                     const ColumnExpression* whereClause = nullptr) 
        -> std::unique_ptr<mlir::ModuleOp>;

    // Helper methods for building MLIR components
    auto registerDialects() -> void;
    auto createRuntimeFunctionDeclarations(mlir::ModuleOp& module) -> void;

private:
    mlir::MLIRContext& context_;
    MLIRLogger* logger_;

    // Internal building methods
    auto buildMainFunction(mlir::ModuleOp& module, const char* tableName, 
                          const std::vector<ColumnExpression>& expressions) -> void;
    auto buildMainFunctionWithWhere(mlir::ModuleOp& module, const char* tableName, 
                                   const std::vector<ColumnExpression>& expressions,
                                   const ColumnExpression* whereClause) -> void;
    auto buildTableScan(const char* tableName) -> mlir::Value;
    auto buildColumnAccess(mlir::OpBuilder& builder, mlir::ModuleOp& module, mlir::Value tupleHandle, const std::vector<ColumnExpression>& expressions) -> void;
};

// Factory function
auto createMLIRBuilder(mlir::MLIRContext& context) -> std::unique_ptr<MLIRBuilder>;

} // namespace mlir_builder

#endif // MLIR_BUILDER_H