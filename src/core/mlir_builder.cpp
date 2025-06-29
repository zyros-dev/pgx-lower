#include "core/mlir_builder.h"
#include "core/mlir_logger.h"
#include "dialects/pg/PgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <sstream>

namespace mlir_builder {

MLIRBuilder::MLIRBuilder(mlir::MLIRContext& context) : context_(context), logger_(nullptr) {
    registerDialects();
}

auto MLIRBuilder::registerDialects() -> void {
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context_.getOrLoadDialect<mlir::pg::PgDialect>();
}

auto MLIRBuilder::buildTableScanModule(const char* tableName, const std::vector<int>& selectedColumns) 
    -> std::unique_ptr<mlir::ModuleOp> {
    
    mlir::OpBuilder builder(&context_);
    auto module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(builder.getUnknownLoc()));
    builder.setInsertionPointToStart(module->getBody());

    createRuntimeFunctionDeclarations(*module);
    buildMainFunction(*module, tableName, selectedColumns);

    return module;
}

auto MLIRBuilder::createRuntimeFunctionDeclarations(mlir::ModuleOp& module) -> void {
    mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToStart(module.getBody());
    
    auto location = builder.getUnknownLoc();
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();

    // Declare external runtime functions as private
    auto funcType = mlir::FunctionType::get(&context_, {i64Type}, {i64Type});
    auto openFunc = builder.create<mlir::func::FuncOp>(location, "open_postgres_table", funcType);
    openFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i64Type}, {i64Type});
    auto readFunc = builder.create<mlir::func::FuncOp>(location, "read_next_tuple_from_table", funcType);
    readFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i64Type}, mlir::TypeRange{});
    auto closeFunc = builder.create<mlir::func::FuncOp>(location, "close_postgres_table", funcType);
    closeFunc.setPrivate();

    // Declare field access functions as private
    funcType = mlir::FunctionType::get(&context_, 
        {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i32Type});
    auto getIntFunc = builder.create<mlir::func::FuncOp>(location, "get_int_field", funcType);
    getIntFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, 
        {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i64Type});
    auto getTextFunc = builder.create<mlir::func::FuncOp>(location, "get_text_field", funcType);
    getTextFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i64Type}, {i1Type});
    auto addTupleFunc = builder.create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    addTupleFunc.setPrivate();
}

auto MLIRBuilder::buildMainFunction(mlir::ModuleOp& module, const char* tableName, 
                                   const std::vector<int>& selectedColumns) -> void {
    mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto location = builder.getUnknownLoc();
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();

    // Create main function
    auto mainFuncType = mlir::FunctionType::get(&context_, {}, {i64Type});
    auto mainFunc = builder.create<mlir::func::FuncOp>(location, "main", mainFuncType);
    auto* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate table open using pg dialect
    auto tableHandleType = mlir::pg::TableHandleType::get(&context_);
    mlir::OperationState scanState(location, mlir::pg::ScanTableOp::getOperationName());
    scanState.addAttribute("table_name", builder.getStringAttr(tableName));
    scanState.addTypes(tableHandleType);
    auto *scanOp = builder.create(scanState);
    auto tableHandle = scanOp->getResult(0);

    // Create iteration loop
    auto zeroConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0));
    auto negTwoConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(-2));
    auto trueConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));

    const auto initialArgs = llvm::SmallVector<mlir::Value>{trueConst, zeroConst};
    const auto argTypes = llvm::SmallVector<mlir::Type>{i1Type, i64Type};

    auto whileOp = builder.create<mlir::scf::WhileOp>(location, argTypes, initialArgs);

    // Before region: condition check
    auto& beforeRegion = whileOp.getBefore();
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, 
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());

    auto continueFlag = beforeRegion.front().getArgument(0);
    builder.create<mlir::scf::ConditionOp>(location, continueFlag, beforeRegion.front().getArguments());

    // After region: read tuple and access fields
    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes, 
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());

    auto tupleCount = afterRegion.front().getArgument(1);

    // Read tuple using pg dialect
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto readState = mlir::OperationState(location, mlir::pg::ReadTupleOp::getOperationName());
    readState.addOperands(tableHandle);
    readState.addTypes(tupleHandleType);
    auto *readOp = builder.create(readState);
    auto tupleHandle = readOp->getResult(0);

    // Convert tuple to i64 for comparison
    auto tupleAsI64 = builder.create<mlir::UnrealizedConversionCastOp>(location, i64Type, tupleHandle).getResult(0);
    auto isEndOfTable = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, 
                                                           tupleAsI64, negTwoConst);

    auto ifOp = builder.create<mlir::scf::IfOp>(location, llvm::ArrayRef<mlir::Type>{i1Type, i64Type}, 
                                               isEndOfTable, true);

    // Then branch: end of table
    auto& thenRegion = ifOp.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    auto falseConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
    auto thenYieldOperands = llvm::SmallVector<mlir::Value>{falseConst, tupleCount};
    builder.create<mlir::scf::YieldOp>(location, thenYieldOperands);

    // Else branch: process tuple with field access
    auto& elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto trueContinue = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));

    // Access each selected column
    buildColumnAccess(tupleHandle, selectedColumns);

    // Output the tuple
    auto addOperands = llvm::SmallVector<mlir::Value> {tupleAsI64};
    auto closeFunc = module.lookupSymbol<mlir::func::FuncOp>("add_tuple_to_result");
    auto addTupleCall = builder.create<mlir::func::CallOp>(location, closeFunc, addOperands);

    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(location, tupleCount, oneIntConst);

    auto elseYieldOperands = llvm::SmallVector<mlir::Value>{trueContinue, newCount.getResult()};
    builder.create<mlir::scf::YieldOp>(location, elseYieldOperands);

    // Continue after the while loop
    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::scf::YieldOp>(location, ifOp.getResults());

    builder.setInsertionPointAfter(whileOp);

    // Close table
    auto tableHandleAsInt = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0));
    auto closeOperands = llvm::SmallVector<mlir::Value> {tableHandleAsInt};
    auto closeFuncOp = module.lookupSymbol<mlir::func::FuncOp>("close_postgres_table");
    builder.create<mlir::func::CallOp>(location, closeFuncOp, closeOperands);

    auto finalCount = whileOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(location, finalCount);
}

auto MLIRBuilder::buildColumnAccess(mlir::Value tupleHandle, const std::vector<int>& selectedColumns) -> void {
    mlir::OpBuilder builder(&context_);
    auto location = builder.getUnknownLoc();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();

    // Access each selected column by its actual index
    for (int columnIndex : selectedColumns) {
        // Determine field type - for now, assume integer types for simplicity
        mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
        getFieldState.addOperands(tupleHandle);
        getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(columnIndex));
        getFieldState.addTypes({i32Type, i1Type}); // value and null flag
        auto getFieldOp = builder.create(getFieldState);
        auto fieldValue = getFieldOp->getResult(0);
        auto fieldNullFlag = getFieldOp->getResult(1);

        // For now, we just access the fields to demonstrate correct indexing
        // The actual tuple output still happens via add_tuple_to_result
    }
}

auto createMLIRBuilder(mlir::MLIRContext& context) -> std::unique_ptr<MLIRBuilder> {
    return std::make_unique<MLIRBuilder>(context);
}

} // namespace mlir_builder