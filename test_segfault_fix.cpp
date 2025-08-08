#include <iostream>
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

int main() {
    std::cerr << "Starting DSA type fix test\n";
    
    mlir::MLIRContext context;
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    std::cerr << "Creating BaseTableOp and MaterializeOp\n";
    
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    std::cerr << "Running RelAlgToDB pass\n";
    
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    
    if (succeeded(result)) {
        std::cerr << "Pass succeeded - now testing MLIR printing\n";
        
        // Test printing DSA types explicitly
        std::string output;
        llvm::raw_string_ostream stream(output);
        funcOp.print(stream);
        std::cerr << "MLIR printing successful\!\n";
        std::cerr << "Output length: " << output.size() << " characters\n";
    } else {
        std::cerr << "Pass failed\n";
        return 1;
    }
    
    std::cerr << "About to destroy module\n";
    module->destroy();
    std::cerr << "Module destroyed - test complete\!\n";
    
    return 0;
}
EOF < /dev/null
