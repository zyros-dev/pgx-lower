#ifndef MLIR_BUILDER_H
#define MLIR_BUILDER_H

#include "mlir_logger.h"
#include <vector>
#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class Value;
}

namespace mlir_builder {

class MLIRBuilder {
public:
    explicit MLIRBuilder(mlir::MLIRContext& context);
    ~MLIRBuilder() = default;

    // PostgreSQL Plan Node to MLIR conversion
    auto buildTableScanModule(const char* tableName, const std::vector<int>& selectedColumns) 
        -> std::unique_ptr<mlir::ModuleOp>;

    // Helper methods for building MLIR components
    auto registerDialects() -> void;
    auto createRuntimeFunctionDeclarations(mlir::ModuleOp& module) -> void;

private:
    mlir::MLIRContext& context_;
    MLIRLogger* logger_;

    // Internal building methods
    auto buildMainFunction(mlir::ModuleOp& module, const char* tableName, 
                          const std::vector<int>& selectedColumns) -> void;
    auto buildTableScan(const char* tableName) -> mlir::Value;
    auto buildColumnAccess(mlir::Value tupleHandle, const std::vector<int>& selectedColumns) -> void;
};

// Factory function
auto createMLIRBuilder(mlir::MLIRContext& context) -> std::unique_ptr<MLIRBuilder>;

} // namespace mlir_builder

#endif // MLIR_BUILDER_H