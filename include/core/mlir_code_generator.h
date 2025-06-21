#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace pgx_lower {

/**
 * Base class for modular MLIR code generation components
 */
class MLIRCodeGenerator {
public:
    MLIRCodeGenerator(mlir::MLIRContext* context, mlir::OpBuilder& builder) 
        : context_(context), builder_(builder) {}
    
    virtual ~MLIRCodeGenerator() = default;

protected:
    mlir::MLIRContext* context_;
    mlir::OpBuilder& builder_;
};

/**
 * Generates table scan operations
 */
class TableScanGenerator : public MLIRCodeGenerator {
public:
    using MLIRCodeGenerator::MLIRCodeGenerator;
    
    // Generate table open call
    mlir::Value generateTableOpen(const std::string& tableName);
    
    // Generate table close call
    void generateTableClose(mlir::Value tableHandle);
    
    // Generate next tuple read
    mlir::Value generateNextTupleRead(mlir::Value tableHandle);
};

/**
 * Generates control flow operations (loops, conditionals)
 */
class ControlFlowGenerator : public MLIRCodeGenerator {
public:
    using MLIRCodeGenerator::MLIRCodeGenerator;
    
    // Generate while loop for tuple iteration
    mlir::scf::WhileOp generateTupleIterationLoop(
        mlir::Value tableHandle,
        mlir::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::Value)> bodyBuilder
    );
    
    // Generate end-of-table condition check
    mlir::Value generateEndOfTableCheck(mlir::Value tupleValue);
};

/**
 * Generates result operations (tuple output)
 */
class ResultGenerator : public MLIRCodeGenerator {
public:
    using MLIRCodeGenerator::MLIRCodeGenerator;
    
    // Generate tuple output call
    mlir::Value generateTupleOutput(mlir::Value tupleValue);
    
    // Generate counter increment
    mlir::Value generateCounterIncrement(mlir::Value currentCount);
};

/**
 * Main orchestrator for modular MLIR generation
 */
class ModularMLIRGenerator {
public:
    ModularMLIRGenerator(mlir::MLIRContext* context);
    
    // Generate complete table scan function
    mlir::func::FuncOp generateTableScanFunction(const std::string& tableName);
    
private:
    mlir::MLIRContext* context_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    std::unique_ptr<TableScanGenerator> tableScanGen_;
    std::unique_ptr<ControlFlowGenerator> controlFlowGen_;
    std::unique_ptr<ResultGenerator> resultGen_;
};

} // namespace pgx_lower