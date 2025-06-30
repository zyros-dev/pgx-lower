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

auto MLIRBuilder::buildTableScanModule(const char* tableName, const std::vector<ColumnExpression>& expressions) 
    -> std::unique_ptr<mlir::ModuleOp> {
    
    mlir::OpBuilder builder(&context_);
    auto module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(builder.getUnknownLoc()));
    builder.setInsertionPointToStart(module->getBody());

    createRuntimeFunctionDeclarations(*module);
    buildMainFunction(*module, tableName, expressions);

    return module;
}

auto MLIRBuilder::buildTableScanModuleWithWhere(const char* tableName, const std::vector<ColumnExpression>& expressions,
                                               const ColumnExpression* whereClause) 
    -> std::unique_ptr<mlir::ModuleOp> {
    
    mlir::OpBuilder builder(&context_);
    auto module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(builder.getUnknownLoc()));
    builder.setInsertionPointToStart(module->getBody());

    createRuntimeFunctionDeclarations(*module);
    buildMainFunctionWithWhere(*module, tableName, expressions, whereClause);

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

    // Declare result storage functions for computed expressions
    funcType = mlir::FunctionType::get(&context_, {i32Type, i32Type, i1Type}, mlir::TypeRange{});
    auto storeIntFunc = builder.create<mlir::func::FuncOp>(location, "store_int_result", funcType);
    storeIntFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i32Type, i1Type, i1Type}, mlir::TypeRange{});
    auto storeBoolFunc = builder.create<mlir::func::FuncOp>(location, "store_bool_result", funcType);
    storeBoolFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i32Type, i64Type, i1Type}, mlir::TypeRange{});
    auto storeBigintFunc = builder.create<mlir::func::FuncOp>(location, "store_bigint_result", funcType);
    storeBigintFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context_, {i32Type}, mlir::TypeRange{});
    auto prepareResultsFunc = builder.create<mlir::func::FuncOp>(location, "prepare_computed_results", funcType);
    prepareResultsFunc.setPrivate();
}

auto MLIRBuilder::buildMainFunction(mlir::ModuleOp& module, const char* tableName, 
                                   const std::vector<ColumnExpression>& expressions) -> void {
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
    buildColumnAccess(builder, module, tupleHandle, expressions);

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

auto MLIRBuilder::buildMainFunctionWithWhere(mlir::ModuleOp& module, const char* tableName, 
                                            const std::vector<ColumnExpression>& expressions,
                                            const ColumnExpression* whereClause) -> void {
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

    // Else branch: process tuple with WHERE clause evaluation
    auto& elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto trueContinue = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));

    // Evaluate WHERE clause predicate if present
    mlir::Value predicateResult = trueConst;
    if (whereClause) {
        // Generate MLIR for WHERE clause predicate
        if (!whereClause->operatorName.empty() && whereClause->operandColumns.size() >= 2) {
            std::vector<mlir::Value> operandValues;
            
            // Load operand columns for predicate
            for (int colIndex : whereClause->operandColumns) {
                if (colIndex >= 0) {
                    mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
                    getFieldState.addOperands(tupleHandle);
                    getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(colIndex));
                    getFieldState.addTypes({i32Type, i1Type});
                    auto getFieldOp = builder.create(getFieldState);
                    operandValues.push_back(getFieldOp->getResult(0));
                }
            }
            
            // Generate comparison operation for WHERE clause
            if (operandValues.size() >= 2) {
                if (whereClause->operatorName == "=") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, operandValues[0], operandValues[1]);
                } else if (whereClause->operatorName == "!=" || whereClause->operatorName == "<>") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ne, operandValues[0], operandValues[1]);
                } else if (whereClause->operatorName == "<") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::slt, operandValues[0], operandValues[1]);
                } else if (whereClause->operatorName == "<=") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sle, operandValues[0], operandValues[1]);
                } else if (whereClause->operatorName == ">") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sgt, operandValues[0], operandValues[1]);
                } else if (whereClause->operatorName == ">=") {
                    predicateResult = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sge, operandValues[0], operandValues[1]);
                }
            }
        }
    }

    // Create conditional execution based on WHERE clause predicate
    auto conditionalIfOp = builder.create<mlir::scf::IfOp>(location, llvm::ArrayRef<mlir::Type>{i1Type, i64Type}, 
                                                          predicateResult, true);

    // Then branch: WHERE clause is true - process the tuple
    auto& whereeThenRegion = conditionalIfOp.getThenRegion();
    if (whereeThenRegion.empty()) {
        builder.createBlock(&whereeThenRegion);
    }
    builder.setInsertionPointToStart(&whereeThenRegion.front());

    // Access selected columns and evaluate expressions
    buildColumnAccess(builder, module, tupleHandle, expressions);

    // Output the tuple
    auto addOperands = llvm::SmallVector<mlir::Value> {tupleAsI64};
    auto addTupleFunc = module.lookupSymbol<mlir::func::FuncOp>("add_tuple_to_result");
    auto addTupleCall = builder.create<mlir::func::CallOp>(location, addTupleFunc, addOperands);

    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(location, tupleCount, oneIntConst);

    auto whereThenYieldOperands = llvm::SmallVector<mlir::Value>{trueContinue, newCount.getResult()};
    builder.create<mlir::scf::YieldOp>(location, whereThenYieldOperands);

    // Else branch: WHERE clause is false - skip the tuple
    auto& whereElseRegion = conditionalIfOp.getElseRegion();
    if (whereElseRegion.empty()) {
        builder.createBlock(&whereElseRegion);
    }
    builder.setInsertionPointToStart(&whereElseRegion.front());

    // Don't process the tuple, just continue with same count
    auto whereElseYieldOperands = llvm::SmallVector<mlir::Value>{trueContinue, tupleCount};
    builder.create<mlir::scf::YieldOp>(location, whereElseYieldOperands);

    // Continue after the WHERE conditional
    builder.setInsertionPointAfter(conditionalIfOp);
    builder.create<mlir::scf::YieldOp>(location, conditionalIfOp.getResults());

    // Continue after the main if op
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

auto MLIRBuilder::buildColumnAccess(mlir::OpBuilder& builder, mlir::ModuleOp& module, mlir::Value tupleHandle, const std::vector<ColumnExpression>& expressions) -> void {
    auto location = builder.getUnknownLoc();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();

    // Count computed expressions to prepare storage
    int computedExpressionCount = 0;
    for (const auto& expr : expressions) {
        if (expr.columnIndex == -1) {
            computedExpressionCount++;
        }
    }
    
    // Prepare storage for computed results if we have any
    if (computedExpressionCount > 0) {
        auto prepareFunc = module.lookupSymbol<mlir::func::FuncOp>("prepare_computed_results");
        auto numComputedConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI32IntegerAttr(expressions.size()));
        builder.create<mlir::func::CallOp>(location, prepareFunc, mlir::ValueRange{numComputedConst});
    }

    // Access each selected column or evaluate expressions
    int resultColumnIndex = 0;
    for (const auto& expr : expressions) {
        if (expr.columnIndex >= 0) {
            // Regular column access
            mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
            getFieldState.addOperands(tupleHandle);
            getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(expr.columnIndex));
            getFieldState.addTypes({i32Type, i1Type}); // value and null flag
            auto getFieldOp = builder.create(getFieldState);
            auto fieldValue = getFieldOp->getResult(0);
            auto fieldNullFlag = getFieldOp->getResult(1);
            
            // For now, we just access the fields to demonstrate correct indexing
            // The actual tuple output still happens via add_tuple_to_result
        } else if (expr.columnIndex == -1) {
            // Computed expression - generate MLIR for arithmetic operations
            if (!expr.operatorName.empty()) {
                std::vector<mlir::Value> operandValues;
                
                // Generate MLIR values for each operand
                for (size_t i = 0; i < expr.operandColumns.size(); ++i) {
                    int colIndex = expr.operandColumns[i];
                    if (colIndex >= 0) {
                        // Load from column
                        mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
                        getFieldState.addOperands(tupleHandle);
                        getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(colIndex));
                        getFieldState.addTypes({i32Type, i1Type});
                        auto getFieldOp = builder.create(getFieldState);
                        operandValues.push_back(getFieldOp->getResult(0));
                    } else if (i < expr.operandConstants.size()) {
                        // Use constant value
                        auto constOp = builder.create<mlir::arith::ConstantOp>(location, 
                                                     builder.getI32IntegerAttr(expr.operandConstants[i]));
                        operandValues.push_back(constOp);
                    }
                }
                
                // Generate the arithmetic operation
                if (operandValues.size() >= 2) {
                    mlir::Value result;
                    if (expr.operatorName == "+") {
                        result = builder.create<mlir::arith::AddIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "-") {
                        result = builder.create<mlir::arith::SubIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "*") {
                        result = builder.create<mlir::arith::MulIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "/") {
                        result = builder.create<mlir::arith::DivSIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "%") {
                        result = builder.create<mlir::arith::RemSIOp>(location, operandValues[0], operandValues[1]);
                    // Comparison operations (return i1 boolean results)
                    } else if (expr.operatorName == "=") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "!=" || expr.operatorName == "<>") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ne, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "<") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::slt, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "<=") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sle, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == ">") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sgt, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == ">=") {
                        result = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sge, operandValues[0], operandValues[1]);
                    // Logical operations (operate on i1 boolean values)
                    } else if (expr.operatorName == "AND" && operandValues.size() >= 2) {
                        result = builder.create<mlir::arith::AndIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "OR" && operandValues.size() >= 2) {
                        result = builder.create<mlir::arith::OrIOp>(location, operandValues[0], operandValues[1]);
                    } else if (expr.operatorName == "NOT" && operandValues.size() >= 1) {
                        // For NOT, we need to XOR with true (1) to flip the boolean
                        auto trueConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));
                        result = builder.create<mlir::arith::XOrIOp>(location, operandValues[0], trueConst);
                    // NULL handling operations (operate on values with null flags)
                    } else if (expr.operatorName == "IS NULL" && expr.operandColumns.size() >= 1) {
                        // For IS NULL, we need to check the null flag returned by GetIntFieldOp
                        int colIndex = expr.operandColumns[0];
                        if (colIndex >= 0) {
                            mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
                            getFieldState.addOperands(tupleHandle);
                            getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(colIndex));
                            getFieldState.addTypes({i32Type, i1Type});
                            auto getFieldOp = builder.create(getFieldState);
                            // Return the null flag (second result)
                            result = getFieldOp->getResult(1);
                        } else {
                            auto falseConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
                            result = falseConst;
                        }
                    } else if (expr.operatorName == "IS NOT NULL" && expr.operandColumns.size() >= 1) {
                        // For IS NOT NULL, we negate the null flag
                        int colIndex = expr.operandColumns[0];
                        if (colIndex >= 0) {
                            mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
                            getFieldState.addOperands(tupleHandle);
                            getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(colIndex));
                            getFieldState.addTypes({i32Type, i1Type});
                            auto getFieldOp = builder.create(getFieldState);
                            // Return the negated null flag
                            auto trueConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));
                            result = builder.create<mlir::arith::XOrIOp>(location, getFieldOp->getResult(1), trueConst);
                        } else {
                            auto trueConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));
                            result = trueConst;
                        }
                    // Aggregate functions (require special handling for multi-row processing)
                    } else if (expr.operatorName == "sum" && operandValues.size() >= 1) {
                        // Placeholder: SUM requires accumulation across multiple rows
                        result = operandValues[0]; // TODO: Implement proper SUM accumulation
                    } else if (expr.operatorName == "avg" && operandValues.size() >= 1) {
                        // Placeholder: AVG requires sum and count across multiple rows
                        result = operandValues[0]; // TODO: Implement proper AVG calculation
                    } else if (expr.operatorName == "count" && operandValues.size() >= 1) {
                        // Placeholder: COUNT requires counting non-null values
                        auto oneConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI32IntegerAttr(1));
                        result = oneConst; // TODO: Implement proper COUNT logic
                    } else if (expr.operatorName == "min" && operandValues.size() >= 1) {
                        // Placeholder: MIN requires comparison across multiple rows
                        result = operandValues[0]; // TODO: Implement proper MIN comparison
                    } else if (expr.operatorName == "max" && operandValues.size() >= 1) {
                        // Placeholder: MAX requires comparison across multiple rows
                        result = operandValues[0]; // TODO: Implement proper MAX comparison
                    // Text operations (operate on string values)
                    } else if (expr.operatorName == "||" && operandValues.size() >= 2) {
                        // String concatenation via runtime function call
                        auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                        auto concatFuncType = mlir::FunctionType::get(
                            builder.getContext(),
                            {ptrType, ptrType}, // Two string inputs
                            {ptrType}           // String output
                        );
                        auto concatFunc = builder.create<mlir::func::CallOp>(
                            location, 
                            "concatenate_strings", 
                            concatFuncType.getResults(),
                            mlir::ValueRange{operandValues[0], operandValues[1]}
                        );
                        result = concatFunc.getResult(0);
                    } else if (expr.operatorName == "~~" && operandValues.size() >= 2) {
                        // LIKE pattern matching via runtime function call
                        auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                        auto likeFuncType = mlir::FunctionType::get(
                            builder.getContext(),
                            {ptrType, ptrType}, // String and pattern inputs
                            {i1Type}            // Boolean output
                        );
                        auto likeFunc = builder.create<mlir::func::CallOp>(
                            location,
                            "string_like_match",
                            likeFuncType.getResults(),
                            mlir::ValueRange{operandValues[0], operandValues[1]}
                        );
                        result = likeFunc.getResult(0);
                    } else if (expr.operatorName == "substring" && operandValues.size() >= 3) {
                        // Substring extraction via runtime function call
                        auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                        auto substringFuncType = mlir::FunctionType::get(
                            builder.getContext(),
                            {ptrType, i32Type, i32Type}, // String, start, length
                            {ptrType}                     // String output
                        );
                        auto substringFunc = builder.create<mlir::func::CallOp>(
                            location,
                            "extract_substring",
                            substringFuncType.getResults(), 
                            mlir::ValueRange{operandValues[0], operandValues[1], operandValues[2]}
                        );
                        result = substringFunc.getResult(0);
                    } else if (expr.operatorName == "upper" && operandValues.size() >= 1) {
                        // String to uppercase via runtime function call
                        auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                        auto upperFuncType = mlir::FunctionType::get(
                            builder.getContext(),
                            {ptrType}, // String input
                            {ptrType}  // String output
                        );
                        auto upperFunc = builder.create<mlir::func::CallOp>(
                            location,
                            "string_to_upper",
                            upperFuncType.getResults(),
                            mlir::ValueRange{operandValues[0]}
                        );
                        result = upperFunc.getResult(0);
                    } else if (expr.operatorName == "lower" && operandValues.size() >= 1) {
                        // String to lowercase via runtime function call
                        auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                        auto lowerFuncType = mlir::FunctionType::get(
                            builder.getContext(),
                            {ptrType}, // String input
                            {ptrType}  // String output
                        );
                        auto lowerFunc = builder.create<mlir::func::CallOp>(
                            location,
                            "string_to_lower",
                            lowerFuncType.getResults(),
                            mlir::ValueRange{operandValues[0]}
                        );
                        result = lowerFunc.getResult(0);
                    // Special operators and functions
                    } else if (expr.operatorName == "COALESCE" && expr.operandColumns.size() >= 1) {
                        // COALESCE returns first non-null value using scf.if chains
                        mlir::Value currentResult;
                        bool firstOperand = true;
                        
                        for (int colIndex : expr.operandColumns) {
                            if (colIndex >= 0) {
                                // Get field value and null flag
                                mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
                                getFieldState.addOperands(tupleHandle);
                                getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(colIndex));
                                getFieldState.addTypes({i32Type, i1Type});
                                auto getFieldOp = builder.create(getFieldState);
                                auto fieldValue = getFieldOp->getResult(0);
                                auto fieldNullFlag = getFieldOp->getResult(1);
                                
                                if (firstOperand) {
                                    // First operand becomes the current result
                                    currentResult = fieldValue;
                                    firstOperand = false;
                                } else {
                                    // Create scf.if to check if current result is null
                                    // If null, use this field value; otherwise keep current result
                                    auto ifOp = builder.create<mlir::scf::IfOp>(location, i32Type, fieldNullFlag, true);
                                    
                                    // Then region: current result is null, use this field
                                    auto& thenRegion = ifOp.getThenRegion();
                                    builder.createBlock(&thenRegion);
                                    builder.setInsertionPointToStart(&thenRegion.front());
                                    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{fieldValue});
                                    
                                    // Else region: current result is not null, keep it
                                    auto& elseRegion = ifOp.getElseRegion();
                                    builder.createBlock(&elseRegion);
                                    builder.setInsertionPointToStart(&elseRegion.front());
                                    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{currentResult});
                                    
                                    currentResult = ifOp.getResult(0);
                                }
                            }
                        }
                        
                        result = currentResult ? currentResult : operandValues[0];
                    } else if (expr.operatorName == "cast" && operandValues.size() >= 1) {
                        // Type cast - placeholder implementation
                        // TODO: Implement proper type conversion in MLIR
                        result = operandValues[0]; // Placeholder: return original value
                    }
                    
                    // Store the computed result for this expression
                    if (result) {
                        auto columnIndexConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI32IntegerAttr(resultColumnIndex));
                        auto falseNullFlag = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
                        
                        // Determine storage function based on result type
                        if (result.getType().isInteger(1)) {
                            // Boolean result (i1) - for comparison and logical operators
                            auto storeBoolFunc = module.lookupSymbol<mlir::func::FuncOp>("store_bool_result");
                            builder.create<mlir::func::CallOp>(location, storeBoolFunc, 
                                mlir::ValueRange{columnIndexConst, result, falseNullFlag});
                        } else if (result.getType().isInteger(32)) {
                            // Integer result (i32) - for arithmetic operators
                            auto storeIntFunc = module.lookupSymbol<mlir::func::FuncOp>("store_int_result");
                            builder.create<mlir::func::CallOp>(location, storeIntFunc, 
                                mlir::ValueRange{columnIndexConst, result, falseNullFlag});
                        } else if (result.getType().isInteger(64)) {
                            // BigInt result (i64) - for large integer operations
                            auto storeBigintFunc = module.lookupSymbol<mlir::func::FuncOp>("store_bigint_result");
                            builder.create<mlir::func::CallOp>(location, storeBigintFunc, 
                                mlir::ValueRange{columnIndexConst, result, falseNullFlag});
                        }
                    }
                }
            }
        }
        resultColumnIndex++;
    }
}

auto createMLIRBuilder(mlir::MLIRContext& context) -> std::unique_ptr<MLIRBuilder> {
    return std::make_unique<MLIRBuilder>(context);
}

} // namespace mlir_builder