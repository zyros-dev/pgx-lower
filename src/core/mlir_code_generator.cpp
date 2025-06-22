#include "core/mlir_code_generator.h"
#include "core/mlir_logger.h"
#include "dialects/pg/PgDialect.h"
#include "dialects/pg/LowerPgToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

namespace pgx_lower {

// ===== TableScanGenerator =====

mlir::Value TableScanGenerator::generateTableOpen(const std::string& tableName) {
    auto location = builder_.getUnknownLoc();
    
    // Use high-level pg.scan_table operation instead of low-level function call
    mlir::OperationState state(location, mlir::pg::ScanTableOp::getOperationName());
    mlir::pg::ScanTableOp::build(builder_, state, tableName);
    
    auto scanOp = builder_.create(state);
    return scanOp->getResult(0);
}

void TableScanGenerator::generateTableClose(mlir::Value tableHandle) {
    auto location = builder_.getUnknownLoc();
    
    builder_.create<mlir::func::CallOp>(
        location,
        mlir::TypeRange{},
        "close_postgres_table",
        mlir::ValueRange{tableHandle}
    );
}

mlir::Value TableScanGenerator::generateNextTupleRead(mlir::Value tableHandle) {
    auto location = builder_.getUnknownLoc();
    
    // Use high-level pg.read_tuple operation instead of direct function call
    mlir::OperationState state(location, mlir::pg::ReadTupleOp::getOperationName());
    mlir::pg::ReadTupleOp::build(builder_, state, tableHandle, {});
    
    auto readOp = builder_.create(state);
    return readOp->getResult(0);
}

// ===== ControlFlowGenerator =====

mlir::scf::WhileOp ControlFlowGenerator::generateTupleIterationLoop(
    mlir::Value tableHandle,
    mlir::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::Value)> bodyBuilder) {
    
    auto location = builder_.getUnknownLoc();
    auto i1Type = builder_.getI1Type();
    auto i64Type = builder_.getI64Type();
    
    // Initial values: continue=true, counter=0
    auto trueVal = builder_.create<mlir::arith::ConstantOp>(
        location, builder_.getBoolAttr(true));
    auto zeroCounter = builder_.create<mlir::arith::ConstantOp>(
        location, builder_.getI64IntegerAttr(0));
    
    auto whileOp = builder_.create<mlir::scf::WhileOp>(
        location,
        mlir::TypeRange{i1Type, i64Type},
        mlir::ValueRange{trueVal, zeroCounter}
    );
    
    // Before region (condition)
    auto& beforeRegion = whileOp.getBefore();
    auto* beforeBlock = builder_.createBlock(&beforeRegion, beforeRegion.end(),
                                             {i1Type, i64Type}, {location, location});
    
    builder_.setInsertionPointToStart(beforeBlock);
    auto continueFlag = beforeBlock->getArgument(0);
    auto counter = beforeBlock->getArgument(1);
    
    builder_.create<mlir::scf::ConditionOp>(location, continueFlag, 
                                            mlir::ValueRange{continueFlag, counter});
    
    // After region (body)
    auto& afterRegion = whileOp.getAfter();
    auto* afterBlock = builder_.createBlock(&afterRegion, afterRegion.end(),
                                            {i1Type, i64Type}, {location, location});
    
    builder_.setInsertionPointToStart(afterBlock);
    continueFlag = afterBlock->getArgument(0);
    counter = afterBlock->getArgument(1);
    
    if (bodyBuilder) {
        bodyBuilder(builder_, location, counter);
    }
    
    return whileOp;
}

mlir::Value ControlFlowGenerator::generateEndOfTableCheck(mlir::Value tupleValue) {
    auto location = builder_.getUnknownLoc();
    
    // Check if tuple value equals -2 (end of table signal)
    auto endSignal = builder_.create<mlir::arith::ConstantOp>(
        location, builder_.getI64IntegerAttr(-2));
    
    auto isEndOfTable = builder_.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::eq, tupleValue, endSignal);
    
    return isEndOfTable;
}

// ===== ResultGenerator =====

mlir::Value ResultGenerator::generateTupleOutput(mlir::Value tupleValue) {
    auto location = builder_.getUnknownLoc();
    auto i1Type = builder_.getI1Type();
    
    auto outputFunc = builder_.create<mlir::func::CallOp>(
        location,
        i1Type,
        "add_tuple_to_result",
        mlir::ValueRange{tupleValue}
    );
    
    return outputFunc.getResult(0);
}

mlir::Value ResultGenerator::generateCounterIncrement(mlir::Value currentCount) {
    auto location = builder_.getUnknownLoc();
    
    auto oneVal = builder_.create<mlir::arith::ConstantOp>(
        location, builder_.getI64IntegerAttr(1));
    
    auto incrementedCount = builder_.create<mlir::arith::AddIOp>(
        location, currentCount, oneVal);
    
    return incrementedCount;
}

// ===== ModularMLIRGenerator =====

ModularMLIRGenerator::ModularMLIRGenerator(mlir::MLIRContext* context) 
    : context_(context) {
    // Register the PostgreSQL dialect
    context->getOrLoadDialect<mlir::pg::PgDialect>();
    
    builder_ = std::make_unique<mlir::OpBuilder>(context);
    tableScanGen_ = std::make_unique<TableScanGenerator>(context, *builder_);
    controlFlowGen_ = std::make_unique<ControlFlowGenerator>(context, *builder_);
    resultGen_ = std::make_unique<ResultGenerator>(context, *builder_);
}

mlir::func::FuncOp ModularMLIRGenerator::generateTableScanFunction(const std::string& tableName) {
    auto location = builder_->getUnknownLoc();
    auto i64Type = builder_->getI64Type();
    
    // Create module if it doesn't exist
    auto moduleOp = builder_->getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        moduleOp = builder_->create<mlir::ModuleOp>(location);
        builder_->setInsertionPointToStart(moduleOp.getBody());
    }
    
    // Declare external functions
    auto funcType = mlir::FunctionType::get(context_, {i64Type}, {i64Type});
    builder_->create<mlir::func::FuncOp>(location, "open_postgres_table", funcType);
    
    funcType = mlir::FunctionType::get(context_, {i64Type}, {i64Type});
    builder_->create<mlir::func::FuncOp>(location, "read_next_tuple_from_table", funcType);
    
    funcType = mlir::FunctionType::get(context_, {i64Type}, {});
    builder_->create<mlir::func::FuncOp>(location, "close_postgres_table", funcType);
    
    funcType = mlir::FunctionType::get(context_, {i64Type}, {builder_->getI1Type()});
    builder_->create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    
    // Create main function
    auto mainFuncType = mlir::FunctionType::get(context_, {}, {i64Type});
    auto mainFunc = builder_->create<mlir::func::FuncOp>(location, "main", mainFuncType);
    
    auto* entryBlock = mainFunc.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);
    
    // Generate table open
    auto tableHandle = tableScanGen_->generateTableOpen(tableName);
    
    // Generate tuple iteration loop
    auto whileOp = controlFlowGen_->generateTupleIterationLoop(
        tableHandle,
        [this, tableHandle](mlir::OpBuilder& loopBuilder, mlir::Location loc, mlir::Value counter) {
            // Read next tuple
            auto tupleValue = tableScanGen_->generateNextTupleRead(tableHandle);
            
            // Check for end of table
            auto isEndOfTable = controlFlowGen_->generateEndOfTableCheck(tupleValue);
            
            // Generate conditional: if end of table, stop; else process tuple
            auto ifOp = loopBuilder.create<mlir::scf::IfOp>(
                loc, 
                mlir::TypeRange{loopBuilder.getI1Type(), loopBuilder.getI64Type()},
                isEndOfTable,
                true // withElseRegion
            );
            
            // Then branch (end of table)
            auto* thenBlock = &ifOp.getThenRegion().front();
            loopBuilder.setInsertionPointToStart(thenBlock);
            auto falseVal = loopBuilder.create<mlir::arith::ConstantOp>(
                loc, loopBuilder.getBoolAttr(false));
            loopBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{falseVal, counter});
            
            // Else branch (process tuple)
            auto* elseBlock = &ifOp.getElseRegion().front();
            loopBuilder.setInsertionPointToStart(elseBlock);
            auto trueVal = loopBuilder.create<mlir::arith::ConstantOp>(
                loc, loopBuilder.getBoolAttr(true));
            
            resultGen_->generateTupleOutput(tupleValue);
            auto newCounter = resultGen_->generateCounterIncrement(counter);
            
            loopBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{trueVal, newCounter});
            
            // Back to loop body
            loopBuilder.setInsertionPointAfter(ifOp);
            loopBuilder.create<mlir::scf::YieldOp>(loc, ifOp.getResults());
        }
    );
    
    // Generate table close
    tableScanGen_->generateTableClose(tableHandle);
    
    // Return final counter
    auto finalCounter = whileOp.getResult(1);
    builder_->create<mlir::func::ReturnOp>(location, finalCounter);
    
    // Log MLIR before lowering to show high-level dialect usage
    std::string beforeLowering;
    llvm::raw_string_ostream beforeStream(beforeLowering);
    mainFunc.print(beforeStream);
    beforeStream.flush();
    
    // Apply lowering pass to convert pg dialect operations to low-level calls
    mlir::PassManager passManager(context_);
    passManager.addNestedPass<mlir::func::FuncOp>(mlir::pg::createLowerPgToSCFPass());
    
    if (moduleOp && mlir::failed(passManager.run(moduleOp))) {
        // Log error but continue - this allows us to see the high-level IR before lowering
        // In production, this would be a proper error
        std::cerr << "Warning: Lowering pass failed, but continuing for debugging\n";
    }
    
    // Log MLIR after lowering to show transformation
    std::string afterLowering;
    llvm::raw_string_ostream afterStream(afterLowering);
    mainFunc.print(afterStream);
    afterStream.flush();
    
    // Only log if there's a difference (indicating pg dialect operations were present)
    if (beforeLowering != afterLowering) {
        // Use simple console output for debugging
        std::cout << "[DEBUG] MLIR before lowering (with pg dialect):\n" << beforeLowering << std::endl;
        std::cout << "[DEBUG] MLIR after lowering (converted to standard dialects):\n" << afterLowering << std::endl;
    }
    
    return mainFunc;
}

} // namespace pgx_lower