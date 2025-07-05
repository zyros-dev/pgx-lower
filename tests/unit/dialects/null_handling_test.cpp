#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <dialects/pg/PgDialect.h>
#include <dialects/pg/LowerPgToSCF.h>
#include <core/mlir_runner.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <llvm/IR/LLVMContext.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/Support/TargetSelect.h>

class NullHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        context_.getOrLoadDialect<mlir::arith::ArithDialect>();
        context_.getOrLoadDialect<mlir::func::FuncDialect>();
        context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context_.getOrLoadDialect<mlir::pg::PgDialect>();
    }

    mlir::MLIRContext context_;
};

TEST_F(NullHandlingTest, CreateGetIntFieldOperation) {
    // Test creating a pg.get_int_field operation and accessing its null indicator
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI1Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_null_indicator", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create a mock tuple handle
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto mockTuple = builder.create<mlir::arith::ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getI64Type(), 12345));
    auto tupleHandle = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, tupleHandleType, mlir::ValueRange{mockTuple}).getResult(0);
    
    // Create pg.get_int_field operation
    auto fieldIndex = builder.getI32IntegerAttr(1);
    auto getFieldOp = builder.create<mlir::pg::GetIntFieldOp>(
        loc, 
        mlir::TypeRange{builder.getI32Type(), builder.getI1Type()},
        tupleHandle, 
        fieldIndex);
    
    // Extract the null indicator (second result)
    auto nullIndicator = getFieldOp.getResult(1);
    
    // Return the null indicator
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{nullIndicator});
    
    // Verify the module is valid
    EXPECT_TRUE(module.verify().succeeded());
    
    // Check that we can access both results
    EXPECT_EQ(getFieldOp.getNumResults(), 2);
    EXPECT_TRUE(getFieldOp.getResult(0).getType().isInteger(32)); // value
    EXPECT_TRUE(getFieldOp.getResult(1).getType().isInteger(1));  // null indicator
}

TEST_F(NullHandlingTest, NullIndicatorDirectUsage) {
    // Test the pattern used in our null handling fix
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI1Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_is_null", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create a mock tuple handle
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto mockTuple = builder.create<mlir::arith::ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getI64Type(), 12345));
    auto tupleHandle = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, tupleHandleType, mlir::ValueRange{mockTuple}).getResult(0);
    
    // Simulate the null test pattern: get field and use null indicator directly
    auto fieldIndex = builder.getI32IntegerAttr(1);
    auto getFieldOp = builder.create<mlir::pg::GetIntFieldOp>(
        loc, 
        mlir::TypeRange{builder.getI32Type(), builder.getI1Type()},
        tupleHandle, 
        fieldIndex);
    
    // For IS NULL: return the null indicator directly
    auto nullIndicator = getFieldOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{nullIndicator});
    
    // Verify the module is valid
    EXPECT_TRUE(module.verify().succeeded());
    
    module.dump();
}

TEST_F(NullHandlingTest, LowerPgToSCFPass) {
    // Test the pg-to-scf lowering pass with null operations
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Add runtime function declarations that the lowering pass expects
    auto ptrType = mlir::LLVM::LLVMPointerType::get(&context_);
    auto getIntFieldType = builder.getFunctionType(
        {ptrType, builder.getI32Type(), ptrType}, 
        {builder.getI32Type()});
    auto getIntFieldFunc = builder.create<mlir::func::FuncOp>(loc, "get_int_field", getIntFieldType);
    getIntFieldFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
    
    auto funcType = builder.getFunctionType({}, {builder.getI1Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_lowering", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create a mock tuple handle
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto mockTuple = builder.create<mlir::arith::ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getI64Type(), 12345));
    auto tupleHandle = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, tupleHandleType, mlir::ValueRange{mockTuple}).getResult(0);
    
    // Create pg.get_int_field operation
    auto fieldIndex = builder.getI32IntegerAttr(1);
    auto getFieldOp = builder.create<mlir::pg::GetIntFieldOp>(
        loc, 
        mlir::TypeRange{builder.getI32Type(), builder.getI1Type()},
        tupleHandle, 
        fieldIndex);
    
    auto nullIndicator = getFieldOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{nullIndicator});
    
    // Verify before lowering
    EXPECT_TRUE(module.verify().succeeded());
    
    std::cout << "Before lowering:\n";
    module.dump();
    
    // Apply pg-to-scf lowering pass
    mlir::PassManager pm(&context_);
    pm.addPass(mlir::pg::createLowerPgToSCFPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(result.succeeded()) << "pg-to-scf lowering pass failed";
    
    std::cout << "\nAfter lowering:\n";
    module.dump();
    
    // Verify after lowering
    EXPECT_TRUE(module.verify().succeeded());
}

TEST_F(NullHandlingTest, LoweringWithUnrealizedConversionCasts) {
    // Test that our lowering handles UnrealizedConversionCast operations properly
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Add runtime function declarations that the lowering pass expects
    auto ptrType = mlir::LLVM::LLVMPointerType::get(&context_);
    auto getIntFieldType = builder.getFunctionType(
        {ptrType, builder.getI32Type(), ptrType}, 
        {builder.getI32Type()});
    auto getIntFieldFunc = builder.create<mlir::func::FuncOp>(loc, "get_int_field", getIntFieldType);
    getIntFieldFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
    
    auto funcType = builder.getFunctionType({}, {builder.getI1Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_unrealized_casts", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create the exact pattern from our null handling implementation
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto mockTuple = builder.create<mlir::arith::ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getI64Type(), 12345));
    
    // This is the UnrealizedConversionCast that might be causing issues
    auto tupleHandle = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, tupleHandleType, mlir::ValueRange{mockTuple}).getResult(0);
    
    auto fieldIndex = builder.getI32IntegerAttr(1);
    auto getFieldOp = builder.create<mlir::pg::GetIntFieldOp>(
        loc, 
        mlir::TypeRange{builder.getI32Type(), builder.getI1Type()},
        tupleHandle, 
        fieldIndex);
    
    auto nullIndicator = getFieldOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{nullIndicator});
    
    // Verify before lowering
    EXPECT_TRUE(module.verify().succeeded());
    
    std::cout << "Module with UnrealizedConversionCast before lowering:\n";
    module.dump();
    
    // Apply pg-to-scf lowering pass
    mlir::PassManager pm(&context_);
    pm.addPass(mlir::pg::createLowerPgToSCFPass());
    
    auto result = pm.run(module);
    
    std::cout << "\nAfter lowering:\n";
    module.dump();
    
    // This test will tell us if the UnrealizedConversionCast is the issue
    if (result.failed()) {
        std::cout << "Pass failed - likely due to UnrealizedConversionCast handling\n";
    } else {
        std::cout << "Pass succeeded - UnrealizedConversionCast not the main issue\n";
    }
    
    // Verify after lowering
    EXPECT_TRUE(module.verify().succeeded());
}
TEST_F(NullHandlingTest, FullPipelineWithLLVMTranslation) {
    // Test the full lowering pipeline including MLIR to LLVM IR translation
    // This mirrors what happens in the PostgreSQL regression tests
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function with the exact pattern from failing regression tests
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Add ALL runtime function declarations that createRuntimeFunctionDeclarations adds
    auto ptrType = mlir::LLVM::LLVMPointerType::get(&context_);
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    
    // Runtime function declarations matching createRuntimeFunctionDeclarations
    auto openTableType = builder.getFunctionType({ptrType}, {ptrType});
    auto openTableFunc = builder.create<mlir::func::FuncOp>(loc, "open_postgres_table", openTableType);
    openTableFunc.setPrivate();
    
    auto readTupleType = builder.getFunctionType({ptrType}, {i64Type});
    auto readTupleFunc = builder.create<mlir::func::FuncOp>(loc, "read_next_tuple_from_table", readTupleType);
    readTupleFunc.setPrivate();
    
    auto getIntFieldType = builder.getFunctionType({ptrType, i32Type, ptrType}, {i32Type});
    auto getIntFieldFunc = builder.create<mlir::func::FuncOp>(loc, "get_int_field", getIntFieldType);
    getIntFieldFunc.setPrivate();
    
    auto storeBoolType = builder.getFunctionType({i32Type, i1Type, i1Type}, mlir::TypeRange{});
    auto storeBoolFunc = builder.create<mlir::func::FuncOp>(loc, "store_bool_result", storeBoolType);
    storeBoolFunc.setPrivate();
    
    // Create main function that mirrors the regression test pattern
    auto mainType = builder.getFunctionType({}, {i64Type});
    auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", mainType);
    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create the exact pattern from failing regression tests
    auto tableHandleType = mlir::pg::TableHandleType::get(&context_);
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    
    // pg.scan_table operation
    auto scanOp = builder.create<mlir::pg::ScanTableOp>(loc, tableHandleType, "current_table");
    
    // pg.read_tuple operation  
    auto readOp = builder.create<mlir::pg::ReadTupleOp>(loc, tupleHandleType, scanOp.getResult());
    
    // Use the tuple handle directly from the read operation
    // The type converter will handle the conversion automatically
    auto tupleHandle = readOp.getResult();
    
    // pg.get_int_field operation with null handling
    auto fieldIndex = builder.getI32IntegerAttr(1);
    auto getFieldOp = builder.create<mlir::pg::GetIntFieldOp>(
        loc, 
        mlir::TypeRange{i32Type, i1Type},
        tupleHandle, 
        fieldIndex);
    
    // Use the null indicator (this is the pattern from our fix)
    auto nullIndicator = getFieldOp.getResult(1);
    
    // Call store_bool_result (pattern from regression tests)
    auto zeroI32 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
    auto falseVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(false));
    builder.create<mlir::func::CallOp>(loc, storeBoolFunc, 
        mlir::ValueRange{zeroI32, nullIndicator, falseVal});
    
    // Return statement
    auto oneI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{oneI64});
    
    // Verify before any lowering
    EXPECT_TRUE(module.verify().succeeded());
    
    std::cout << "=== BEFORE ANY LOWERING ===\n";
    module.dump();
    
    // Apply pg-to-scf lowering pass
    mlir::PassManager pm1(&context_);
    pm1.addPass(mlir::pg::createLowerPgToSCFPass());
    
    auto pgResult = pm1.run(module);
    EXPECT_TRUE(pgResult.succeeded()) << "pg-to-scf lowering pass failed";
    
    std::cout << "\n=== AFTER PG-TO-SCF LOWERING ===\n";
    module.dump();
    
    // Apply standard lowering passes (matching mlir_runner.cpp)
    mlir::PassManager pm2(&context_);
    pm2.addPass(mlir::createConvertSCFToCFPass());
    pm2.addPass(mlir::createArithToLLVMConversionPass());
    pm2.addPass(mlir::createConvertFuncToLLVMPass());
    pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
    
    // Add the cleanup pass that should handle UnrealizedConversionCast
    pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    auto standardResult = pm2.run(module);
    EXPECT_TRUE(standardResult.succeeded()) << "Standard lowering passes failed";
    
    std::cout << "\n=== AFTER ALL LOWERING PASSES ===\n";
    module.dump();
    
    // Check for remaining UnrealizedConversionCast operations
    bool hasUnrealizedCasts = false;
    module.walk([&](mlir::Operation* op) {
        if (auto castOp = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
            std::cout << "FOUND UnrealizedConversionCast: ";
            castOp.dump();
            hasUnrealizedCasts = true;
        }
    });
    
    if (hasUnrealizedCasts) {
        std::cout << "\n❌ UnrealizedConversionCast operations remain - this will cause MLIR to LLVM IR translation to fail\n";
    } else {
        std::cout << "\n✅ No UnrealizedConversionCast operations remain - MLIR to LLVM IR translation should succeed\n";
    }
    
    // Test MLIR to LLVM IR translation manually
    // Register LLVM IR translation before attempting translation
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context_.appendDialectRegistry(registry);
    mlir::registerLLVMDialectTranslation(context_);
    mlir::registerBuiltinDialectTranslation(context_);
    
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
    
    if (llvmModule) {
        std::cout << "✅ MLIR to LLVM IR translation SUCCEEDED\n";
        EXPECT_TRUE(true);
    } else {
        std::cout << "❌ MLIR to LLVM IR translation FAILED - this is the root cause of regression test failures\n";
        EXPECT_TRUE(false) << "MLIR to LLVM IR translation failed";
    }
    
    // Verify final module is still valid
    EXPECT_TRUE(module.verify().succeeded());
}

TEST_F(NullHandlingTest, MinimalJitExecution) {
    // Create the simplest possible MLIR function to test basic JIT execution
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create a function like LingoDB: void main() - no return value (follow their successful pattern)
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Just return void (like LingoDB pattern)
    builder.create<mlir::func::ReturnOp>(loc);
    
    std::cout << "\n=== MINIMAL JIT TEST ===\n";
    module.print(llvm::outs());
    std::cout << "\n";
    
    // Test lowering to LLVM dialect
    mlir::PassManager pm(&context_);
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    
    if (mlir::failed(pm.run(module))) {
        std::cout << "❌ Standard lowering to LLVM failed\n";
        EXPECT_TRUE(false) << "Standard lowering failed";
        return;
    }
    
    std::cout << "✅ Standard lowering to LLVM succeeded\n";
    
    // Test MLIR to LLVM IR translation
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context_.appendDialectRegistry(registry);
    mlir::registerLLVMDialectTranslation(context_);
    mlir::registerBuiltinDialectTranslation(context_);
    
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
    
    if (!llvmModule) {
        std::cout << "❌ MLIR to LLVM IR translation failed\n";
        EXPECT_TRUE(false) << "MLIR to LLVM IR translation failed";
        return;
    }
    
    std::cout << "✅ MLIR to LLVM IR translation succeeded\n";
    
    // Initialize LLVM native target for JIT execution
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    
    // Test JIT execution
    auto maybeEngine = mlir::ExecutionEngine::create(module);
    
    if (!maybeEngine) {
        std::cout << "❌ ExecutionEngine creation failed: " << llvm::toString(maybeEngine.takeError()) << "\n";
        EXPECT_TRUE(false) << "ExecutionEngine creation failed";
        return;
    }
    
    auto engine = std::move(*maybeEngine);
    std::cout << "✅ ExecutionEngine created successfully\n";
    
    // Lookup the main function
    auto mainFunc = engine->lookupPacked("main");
    if (!mainFunc) {
        std::cout << "❌ Failed to lookup main function\n";
        EXPECT_TRUE(false) << "Failed to lookup main function";
        return;
    }
    
    // Use LingoDB pattern: void (*)() function signature
    auto fptr = reinterpret_cast<void (*)()>(mainFunc.get());
    std::cout << "✅ Main function lookup succeeded\n";
    
    // Execute the JIT function
    try {
        std::cout << "🚀 Calling minimal JIT function...\n";
        fptr();  // void function call (LingoDB pattern)
        std::cout << "🎉 JIT execution SUCCESS! Void function executed without crash!\n";
        std::cout << "✅ BASIC JIT MECHANISM WORKS!\n";
    } catch (const std::exception& e) {
        std::cout << "❌ JIT execution failed with exception: " << e.what() << "\n";
        EXPECT_TRUE(false) << "JIT execution failed with exception";
    } catch (...) {
        std::cout << "❌ JIT execution failed with unknown exception\n"; 
        EXPECT_TRUE(false) << "JIT execution failed with unknown exception";
    }
}
