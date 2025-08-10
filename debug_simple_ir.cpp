//===- debug_simple_ir.cpp - Minimal IR construction test ===============//

#include <iostream>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

int main() {
    MLIRContext context;
    context.loadDialect<func::FuncDialect, arith::ArithDialect>();
    
    std::cout << "Creating basic IR..." << std::endl;
    
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Add simple return
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(func);
    
    std::cout << "Testing module walk..." << std::endl;
    
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
    
    std::cout << "Walk completed normally with " << count << " operations." << std::endl;
    return 0;
}