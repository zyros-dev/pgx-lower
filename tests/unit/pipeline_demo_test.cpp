#include <gtest/gtest.h>
#include <iostream>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "core/mlir_logger.h"

// Include all the dialects
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/LowerSubOpToDB.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/LowerDBToDSA.h"
#include "dialects/dsa/DSADialect.h"
#include "dialects/util/LowerDSAToLLVM.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/util/UtilDialect.h"

void printModule(mlir::ModuleOp module, const std::string& stage) {
    std::cout << "\n===========================================" << std::endl;
    std::cout << "=== " << stage << " ===" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    os.flush();
    std::cout << str << std::endl;
}

TEST(PipelineDemo, ShowMLIRTransformations) {
    std::cout << "\n\n*** DEMONSTRATING COMPLETE LINGODB LOWERING PIPELINE ***" << std::endl;
    std::cout << "*** RelAlg → SubOp → DB → DSA → LLVM ***\n" << std::endl;
    
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
    printModule(module, "STAGE 1: Initial RelAlg Dialect");
    
    // Run RelAlg → SubOp lowering
    std::cout << "\n>>> Running RelAlg → SubOp lowering..." << std::endl;
    mlir::PassManager pm1(&context);
    pm1.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
    ASSERT_TRUE(succeeded(pm1.run(module))) << "RelAlg → SubOp lowering failed!";
    printModule(module, "STAGE 2: After RelAlg → SubOp");
    
    // Run SubOp → DB lowering
    std::cout << "\n>>> Running SubOp → DB lowering..." << std::endl;
    mlir::PassManager pm2(&context);
    pm2.addPass(pgx_lower::compiler::dialect::subop::createLowerSubOpPass());
    ASSERT_TRUE(succeeded(pm2.run(module))) << "SubOp → DB lowering failed!";
    printModule(module, "STAGE 3: After SubOp → DB");
    
    // Run DB → DSA lowering
    std::cout << "\n>>> Running DB → DSA lowering..." << std::endl;
    mlir::PassManager pm3(&context);
    pm3.addPass(pgx_lower::compiler::dialect::db::createLowerToStdPass());
    ASSERT_TRUE(succeeded(pm3.run(module))) << "DB → DSA lowering failed!";
    printModule(module, "STAGE 4: After DB → DSA");
    
    // Run DSA → LLVM lowering
    std::cout << "\n>>> Running DSA → LLVM lowering..." << std::endl;
    mlir::PassManager pm4(&context);
    pm4.addPass(pgx_lower::compiler::dialect::util::createUtilToLLVMPass());
    ASSERT_TRUE(succeeded(pm4.run(module))) << "DSA → LLVM lowering failed!";
    printModule(module, "STAGE 5: After DSA → LLVM");
    
    std::cout << "\n\n✓✓✓ COMPLETE LINGODB LOWERING PIPELINE DEMONSTRATED! ✓✓✓" << std::endl;
    std::cout << "    RelAlg → SubOp → DB → DSA → LLVM" << std::endl;
    std::cout << "\nThis shows how a simple SQL query transforms through each dialect layer." << std::endl;
    std::cout << "Each layer handles specific concerns:" << std::endl;
    std::cout << "  - RelAlg: High-level relational operations" << std::endl;
    std::cout << "  - SubOp: Execution steps and tuple processing" << std::endl;
    std::cout << "  - DB: Type polymorphism and NULL handling" << std::endl;
    std::cout << "  - DSA: Data structure access patterns" << std::endl;
    std::cout << "  - LLVM: Final machine code generation\n" << std::endl;
}