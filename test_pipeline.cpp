#include <iostream>
#include <string>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

// Include all the dialects
#include "compiler/Dialect/relalg/RelAlgDialect.h"
#include "compiler/Dialect/relalg/RelAlgOps.h"
#include "compiler/Dialect/relalg/LowerRelAlgToSubOp.h"
#include "compiler/Dialect/subop/SubOpDialect.h"
#include "compiler/Dialect/subop/SubOpOps.h"
#include "compiler/Dialect/subop/LowerSubOpToDB.h"
#include "compiler/Dialect/db/DBDialect.h"
#include "compiler/Dialect/db/DBOps.h"
#include "compiler/Dialect/db/LowerDBToDSA.h"
#include "compiler/Dialect/DSA/DSADialect.h"
#include "compiler/Dialect/DSA/DSAOps.h"
#include "compiler/Dialect/util/LowerDSAToLLVM.h"
#include "compiler/Dialect/tuplestream/TupleStreamDialect.h"
#include "compiler/Dialect/tuplestream/ColumnManager.h"
#include "compiler/Dialect/util/UtilDialect.h"

void printModule(mlir::ModuleOp module, const std::string& stage) {
    std::cout << "\n=== " << stage << " ===" << std::endl;
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    os.flush();
    std::cout << str << std::endl;
}

int main() {
    // Initialize MLIR context with all required dialects
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::relalg::RelAlgDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::subop::SubOpDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::dsa::DSADialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::util::UtilDialect>();
    
    // Create a simple module simulating SELECT * FROM test
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create main function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Get column manager from TupleStream dialect
    auto* tupleStreamDialect = context.getLoadedDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
    auto& columnManager = tupleStreamDialect->getColumnManager();
    columnManager.setContext(&context);
    
    // Create a simple RelAlg query: SELECT * FROM test
    // First, create column definitions for the table
    auto idColumn = columnManager.createDef("test", "id");
    
    // Create table metadata
    auto tableColumns = builder.getArrayAttr({idColumn});
    
    // Create basetable operation
    auto tableOp = builder.create<pgx_lower::compiler::dialect::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("test"),  // table name
        tableColumns                    // columns
    );
    
    // Create return operation (using the table directly)
    builder.create<pgx_lower::compiler::dialect::relalg::ReturnOp>(
        builder.getUnknownLoc(),
        tableOp.getResult()
    );
    
    // Create func.return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Print initial RelAlg representation
    printModule(module, "Initial RelAlg Dialect");
    
    // Run RelAlg → SubOp lowering
    std::cout << "\nRunning RelAlg → SubOp lowering..." << std::endl;
    mlir::PassManager pm1(&context);
    pm1.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
    if (failed(pm1.run(module))) {
        std::cerr << "RelAlg → SubOp lowering failed!" << std::endl;
        return 1;
    }
    printModule(module, "After RelAlg → SubOp");
    
    // Run SubOp → DB lowering
    std::cout << "\nRunning SubOp → DB lowering..." << std::endl;
    mlir::PassManager pm2(&context);
    pm2.addPass(pgx_lower::compiler::dialect::subop::createLowerSubOpPass());
    if (failed(pm2.run(module))) {
        std::cerr << "SubOp → DB lowering failed!" << std::endl;
        return 1;
    }
    printModule(module, "After SubOp → DB");
    
    // Run DB → DSA lowering
    std::cout << "\nRunning DB → DSA lowering..." << std::endl;
    mlir::PassManager pm3(&context);
    pm3.addPass(pgx_lower::compiler::dialect::db::createLowerToStdPass());
    if (failed(pm3.run(module))) {
        std::cerr << "DB → DSA lowering failed!" << std::endl;
        return 1;
    }
    printModule(module, "After DB → DSA");
    
    // Run DSA → LLVM lowering
    std::cout << "\nRunning DSA → LLVM lowering..." << std::endl;
    mlir::PassManager pm4(&context);
    pm4.addPass(pgx_lower::compiler::dialect::util::createUtilToLLVMPass());
    if (failed(pm4.run(module))) {
        std::cerr << "DSA → LLVM lowering failed!" << std::endl;
        return 1;
    }
    printModule(module, "After DSA → LLVM");
    
    std::cout << "\n✓ Complete LingoDB lowering pipeline demonstrated!" << std::endl;
    std::cout << "  RelAlg → SubOp → DB → DSA → LLVM" << std::endl;
    
    return 0;
}