//===- debug_dsa_simple.cpp - Minimal DSA operation test ================//

#include <iostream>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

using namespace mlir;

int main() {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    
    std::cout << "Creating minimal DSA IR..." << std::endl;
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    std::cout << "About to create TableBuilderType..." << std::endl;
    
    // Create TableBuilder type (this might be the issue)
    auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(
        &context, TupleType::get(&context, {builder.getI64Type()}));
    
    std::cout << "TableBuilderType created successfully" << std::endl;
    std::cout << "About to create CreateDS..." << std::endl;
    
    // Create DSA operation (this is where the circular reference might be created)
    auto createDSOp = builder.create<pgx::mlir::dsa::CreateDS>(
        builder.getUnknownLoc(), tableBuilderType, 
        builder.getStringAttr("id:int[64]"));
    
    std::cout << "CreateDS created successfully" << std::endl;
    
    // Add simple return
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    std::cout << "Testing module walk after DSA operation..." << std::endl;
    
    int count = 0;
    module.walk([&](Operation* op) {
        count++;
        std::cout << "Operation " << count << ": " << op->getName().getStringRef().str() << std::endl;
        if (count > 10) {
            std::cout << "STOPPING - potential infinite loop detected" << std::endl;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    
    std::cout << "Walk completed with " << count << " operations." << std::endl;
    return 0;
}