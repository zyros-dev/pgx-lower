// Minimal SubOp to Control Flow pass - just removes ExecutionGroupOp
#include "dialects/subop/SubOpToControlFlow.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}
#endif

#include "dialects/subop/SubOpOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cctype>

namespace pgx_lower::compiler::dialect::subop {

namespace {

class MinimalSubOpToControlFlowPass : public mlir::PassWrapper<MinimalSubOpToControlFlowPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MinimalSubOpToControlFlowPass)
    
    MinimalSubOpToControlFlowPass() = default;
    MinimalSubOpToControlFlowPass(const MinimalSubOpToControlFlowPass&) = default;
    
    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<mlir::scf::SCFDialect>();
    }
    
    llvm::StringRef getArgument() const override { return "minimal-lower-subop"; }
    llvm::StringRef getDescription() const override { return "Minimal SubOp to Control Flow lowering"; }
    
    void runOnOperation() override {
#ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "=== MinimalSubOpToControlFlowPass::runOnOperation() START ===");
#endif
        mlir::ModuleOp module = getOperation();
        mlir::OpBuilder builder(&getContext());
        
        // Find ExecutionGroupOp operations and process them  
        llvm::SmallVector<ExecutionGroupOp, 4> execGroups;
        module.walk([&](ExecutionGroupOp op) {
            execGroups.push_back(op);
        });
        
#ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "Found %d ExecutionGroupOp operations to process", (int)execGroups.size());
#endif
        
        // Extract table information from ExecutionGroupOp operations
        std::string tableName;
        for (auto execGroup : execGroups) {
#ifdef POSTGRESQL_EXTENSION
            elog(NOTICE, "Processing ExecutionGroupOp with %d operations", (int)execGroup.getRegion().front().getOperations().size());
#endif
            // Process each operation in the ExecutionGroupOp
            for (auto& op : execGroup.getRegion().front().getOperations()) {
                if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) {
                    continue; // Skip return operations
                }
                
#ifdef POSTGRESQL_EXTENSION
                elog(NOTICE, "Processing operation: %s", op.getName().getStringRef().data());
#endif
                
                if (auto getExternal = mlir::dyn_cast<subop::GetExternalOp>(op)) {
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "Found GetExternalOp - extracting table name");
#endif
                    // Extract table name from the JSON-like description
                    // Format: { "table": "test", "mapping": { "id$0" :"id"} }
                    auto descr = getExternal.getDescr();
                    std::string descrStr = descr.str();
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "GetExternalOp description: %s", descrStr.c_str());
#endif
                    // Simple extraction - find "table": "xxx"
                    size_t tablePos = descrStr.find("\"table\"");
                    if (tablePos != std::string::npos) {
                        size_t colonPos = descrStr.find(":", tablePos);
                        size_t firstQuote = descrStr.find("\"", colonPos);
                        size_t secondQuote = descrStr.find("\"", firstQuote + 1);
                        if (firstQuote != std::string::npos && secondQuote != std::string::npos) {
                            tableName = descrStr.substr(firstQuote + 1, secondQuote - firstQuote - 1);
#ifdef POSTGRESQL_EXTENSION
                            elog(NOTICE, "Extracted table name: %s", tableName.c_str());
#endif
                        }
                    }
                } else if (auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(op)) {
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "Found ScanRefsOp - table scanning");
#endif
                } else if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) {
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "Found GatherOp - data gathering");
#endif
                } else {
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "Unknown SubOp operation: %s", op.getName().getStringRef().data());
#endif
                }
            }
        }
        
        // Create main function if it doesn't exist
        auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
#ifdef POSTGRESQL_EXTENSION
            elog(NOTICE, "Creating main function - starting with basic version");
#endif
            builder.setInsertionPointToStart(module.getBody());
            // IMPORTANT: invokePacked expects void return type, not i32
            auto funcType = builder.getFunctionType({}, {});
            mainFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "main", funcType);
            mainFunc.setPublic();
            
            // Create entry block with basic PostgreSQL result preparation
            auto* entryBlock = mainFunc.addEntryBlock();
            builder.setInsertionPointToStart(entryBlock);
            
            // Generate actual PostgreSQL table scanning code
            auto i32Type = builder.getI32Type();
            auto i64Type = builder.getI64Type();
            auto indexType = builder.getIndexType();
            
            // Declare PostgreSQL runtime functions
            auto prepareFunc = module.lookupSymbol<mlir::func::FuncOp>("prepare_computed_results");
            if (!prepareFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto prepareFuncType = builder.getFunctionType({i32Type}, {});
                prepareFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "prepare_computed_results", prepareFuncType);
                prepareFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            auto ptrType = mlir::LLVM::LLVMPointerType::get(&getContext());
            
            auto openTableFunc = module.lookupSymbol<mlir::func::FuncOp>("open_postgres_table");
            if (!openTableFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto openTableFuncType = builder.getFunctionType({ptrType}, ptrType);
                openTableFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "open_postgres_table", openTableFuncType);
                openTableFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            auto readNextFunc = module.lookupSymbol<mlir::func::FuncOp>("read_next_tuple_from_table");
            if (!readNextFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto readNextFuncType = builder.getFunctionType({ptrType}, i64Type);
                readNextFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "read_next_tuple_from_table", readNextFuncType);
                readNextFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            auto getIntFieldFunc = module.lookupSymbol<mlir::func::FuncOp>("get_int_field_mlir");
            if (!getIntFieldFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto getIntFieldFuncType = builder.getFunctionType({i64Type, i32Type}, i32Type);
                getIntFieldFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "get_int_field_mlir", getIntFieldFuncType);
                getIntFieldFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            // Add function to store field as datum based on actual type
            auto storeFieldAsDatumFunc = module.lookupSymbol<mlir::func::FuncOp>("store_field_as_datum");
            if (!storeFieldAsDatumFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto storeFieldAsDatumFuncType = builder.getFunctionType({i32Type, i64Type, i32Type}, {});
                storeFieldAsDatumFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "store_field_as_datum", storeFieldAsDatumFuncType);
                storeFieldAsDatumFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            auto storeIntFunc = module.lookupSymbol<mlir::func::FuncOp>("store_int_result");
            if (!storeIntFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                // Fix: store_int_result expects 3 parameters: (int32_t columnIndex, int32_t value, bool isNull)
                auto i1Type = builder.getI1Type();
                auto storeIntFuncType = builder.getFunctionType({i32Type, i32Type, i1Type}, {});
                storeIntFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "store_int_result", storeIntFuncType);
                storeIntFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            // Add function to stream individual tuples
            auto addTupleFunc = module.lookupSymbol<mlir::func::FuncOp>("add_tuple_to_result");
            if (!addTupleFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto addTupleFuncType = builder.getFunctionType({i64Type}, {builder.getI1Type()});
                addTupleFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "add_tuple_to_result", addTupleFuncType);
                addTupleFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            // Add function to mark results ready for streaming (called at end)
            auto markResultsReadyFunc = module.lookupSymbol<mlir::func::FuncOp>("mark_results_ready_for_streaming");
            if (!markResultsReadyFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto markResultsReadyFuncType = builder.getFunctionType({}, {});
                markResultsReadyFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "mark_results_ready_for_streaming", markResultsReadyFuncType);
                markResultsReadyFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            auto closeTableFunc = module.lookupSymbol<mlir::func::FuncOp>("close_postgres_table");
            if (!closeTableFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto closeTableFuncType = builder.getFunctionType({ptrType}, {});
                closeTableFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "close_postgres_table", closeTableFuncType);
                closeTableFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            // Generate PostgreSQL table scanning code with loop for multiple tuples
#ifdef POSTGRESQL_EXTENSION
            elog(NOTICE, "Generating PostgreSQL table scanning code with loop support");
#endif
            
            // Count number of columns from the ExecutionGroupOp
            int numColumns = 1; // Default to 1 column
            
#ifdef POSTGRESQL_EXTENSION
            elog(NOTICE, "MinimalSubOpToControlFlow: Looking for column count in ExecutionGroupOp");
#endif
            
            // Try to determine actual column count from operations
            for (auto execGroup : execGroups) {
                // Look for GetExternalOp to extract column information
                for (auto& op : execGroup.getRegion().front().getOperations()) {
                    if (auto getExternal = mlir::dyn_cast<subop::GetExternalOp>(op)) {
                        // The description contains column mapping information
                        // Count the number of column mappings in the format "col$N":"colname"
                        auto descr = getExternal.getDescr();
                        std::string descrStr = descr.str();
#ifdef POSTGRESQL_EXTENSION
                        elog(NOTICE, "MinimalSubOpToControlFlow: Found GetExternalOp with description: %s", descrStr.c_str());
#endif
                        
                        // Count occurrences of pattern "$N\""
                        numColumns = 0;
                        size_t pos = 0;
                        while ((pos = descrStr.find("$", pos)) != std::string::npos) {
                            // Check if this is a column mapping (has a digit after $)
                            if (pos + 1 < descrStr.length() && std::isdigit(descrStr[pos + 1])) {
                                numColumns++;
                            }
                            pos++;
                        }
                        
                        if (numColumns == 0) {
                            numColumns = 1; // Default to 1 if we can't parse
                        }
                        
#ifdef POSTGRESQL_EXTENSION
                        elog(NOTICE, "Detected %d columns from GetExternalOp description: %s", 
                             numColumns, descrStr.c_str());
#endif
                        break;
                    }
                }
            }
            
            // Create constants
            auto i1Type = builder.getI1Type();
            auto zero32 = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 32);
            auto one32 = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 1, 32);
            auto two32 = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 2, 32);
            auto zero64 = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 64);
            auto falseVal = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 1);
            auto trueVal = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 1, 1);
            
            // Create table name string constant from extracted table name
            mlir::Value tableNamePtr;
            if (!tableName.empty()) {
#ifdef POSTGRESQL_EXTENSION
                elog(NOTICE, "Creating LLVM string constant for table name: %s", tableName.c_str());
#endif
                // Create a global string constant for the table name
                auto stringType = mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), tableName.length() + 1);
                auto stringAttr = builder.getStringAttr(tableName + '\0');
                
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                
                auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
                    module.getLoc(),
                    stringType,
                    /*isConstant=*/true,
                    mlir::LLVM::Linkage::Internal,
                    "table_name_str",
                    stringAttr,
                    /*alignment=*/0);
                
                builder.restoreInsertionPoint(savedIP);
                
                // Get address of the global string
                auto globalAddr = builder.create<mlir::LLVM::AddressOfOp>(
                    module.getLoc(), 
                    mlir::LLVM::LLVMPointerType::get(&getContext()),
                    globalOp.getSymName());
                
                // Cast to generic pointer
                tableNamePtr = builder.create<mlir::LLVM::BitcastOp>(
                    module.getLoc(), ptrType, globalAddr);
            } else {
#ifdef POSTGRESQL_EXTENSION
                elog(WARNING, "No table name extracted, using NULL pointer");
#endif
                tableNamePtr = builder.create<mlir::LLVM::ZeroOp>(module.getLoc(), ptrType);
            }
            
            // Prepare results storage based on number of columns
            auto numColsConst = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), numColumns, 32);
            mlir::Value prepareArgs[] = {numColsConst};
            builder.create<mlir::func::CallOp>(module.getLoc(), prepareFunc, prepareArgs);
            
            // Open the table
            mlir::Value openArgs[] = {tableNamePtr};
            auto tableHandle = builder.create<mlir::func::CallOp>(module.getLoc(), openTableFunc, 
                                                                 openArgs).getResult(0);
            
            // Create loop to read all tuples
            // Use scf.while loop with tuple pointer as loop-carried value
            auto scfBuilder = builder;
            
            // Read first tuple before loop
            mlir::Value initialReadArgs[] = {tableHandle};
            auto initialTuple = scfBuilder.create<mlir::func::CallOp>(module.getLoc(), readNextFunc, 
                                                                    initialReadArgs).getResult(0);
            
            // Create while loop that processes all tuples
            auto whileOp = scfBuilder.create<mlir::scf::WhileOp>(
                module.getLoc(),
                mlir::TypeRange{i64Type}, // Loop-carried value: tuple pointer
                mlir::ValueRange{initialTuple}, // Initial value: first tuple
                [&](mlir::OpBuilder& beforeBuilder, mlir::Location loc, mlir::ValueRange args) {
                    // Before region: check if current tuple is valid
                    auto currentTuple = args[0];
                    
                    // Check if tuple is valid (non-zero)
                    auto isValid = beforeBuilder.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::ne, currentTuple, zero64);
                    
                    // Pass the tuple to the after region
                    beforeBuilder.create<mlir::scf::ConditionOp>(loc, isValid, args);
                },
                [&](mlir::OpBuilder& afterBuilder, mlir::Location loc, mlir::ValueRange args) {
                    // After region: process the current tuple
                    auto currentTuple = args[0];
                    
                    // Extract and store all columns using type-aware function
#ifdef POSTGRESQL_EXTENSION
                    elog(NOTICE, "MinimalSubOpToControlFlow: Processing tuple with %d columns", numColumns);
#endif
                    for (int colIdx = 0; colIdx < numColumns; colIdx++) {
#ifdef POSTGRESQL_EXTENSION
                        elog(NOTICE, "MinimalSubOpToControlFlow: Generating store_field_as_datum call for column %d", colIdx);
#endif
                        // TODO: Get actual field index from column mapping
                        // For now, this assumes we're selecting all columns in order
                        // This breaks for queries like "SELECT char_col, varchar_col, text_col"
                        // where we need to map column 0->9, 1->10, 2->11
                        auto colIdxConst = afterBuilder.create<mlir::arith::ConstantIntOp>(loc, colIdx, 32);
                        mlir::Value storeFieldArgs[] = {colIdxConst, currentTuple, colIdxConst};
                        afterBuilder.create<mlir::func::CallOp>(loc, storeFieldAsDatumFunc, storeFieldArgs);
                    }
                    
                    // Stream this tuple to the output
                    mlir::Value addTupleArgs[] = {currentTuple};
                    afterBuilder.create<mlir::func::CallOp>(loc, addTupleFunc, addTupleArgs);
                    
                    // Read next tuple for next iteration
                    mlir::Value readArgs[] = {tableHandle};
                    auto nextTuple = afterBuilder.create<mlir::func::CallOp>(loc, readNextFunc, 
                                                                            readArgs).getResult(0);
                    
                    afterBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{nextTuple});
                });
            
            // Close the table after loop
            mlir::Value closeArgs[] = {tableHandle};
            builder.create<mlir::func::CallOp>(module.getLoc(), closeTableFunc, closeArgs);
            
            // Mark results ready for streaming (final signal to executor)
            builder.create<mlir::func::CallOp>(module.getLoc(), markResultsReadyFunc, mlir::ValueRange{});
            
            // Return void (no values) since invokePacked expects void function
            builder.create<mlir::func::ReturnOp>(module.getLoc(), mlir::ValueRange{});
        }
        
        // Remove ExecutionGroupOp operations
        for (auto op : execGroups) {
            // Replace all uses with empty tuple stream (placeholder)
            for (auto result : op.getResults()) {
                result.replaceAllUsesWith(mlir::Value{});
            }
            op.erase();
        }
        
#ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "=== MinimalSubOpToControlFlowPass::runOnOperation() COMPLETE ===");
#endif
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createMinimalSubOpToControlFlowPass() {
    return std::make_unique<MinimalSubOpToControlFlowPass>();
}

} // namespace pgx_lower::compiler::dialect::subop