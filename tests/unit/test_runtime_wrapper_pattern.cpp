#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

// Test to understand the runtime wrapper pattern used in LingoDB
class RuntimeWrapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<LLVM::LLVMDialect>();
        context.loadDialect<func::FuncDialect>();
    }

    MLIRContext context;
};

// Simulating the LingoDB pattern where rt::Function::method(builder, loc) returns a callable
namespace test_runtime {

// This mimics what LingoDB appears to do
struct CallableWrapper {
    OpBuilder& builder;
    Location loc;
    std::string funcName;
    
    CallableWrapper(const std::string& name, OpBuilder& b, Location l) 
        : builder(b), loc(l), funcName(name) {}
    
    // The operator() allows ({args}) syntax
    std::vector<Value> operator()(std::initializer_list<Value> args) {
        // In real implementation, this would generate MLIR call
        auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
        
        // Create or find the function
        auto funcOp = module.lookupSymbol<func::FuncOp>(funcName);
        if (!funcOp) {
            // Create function signature based on args
            SmallVector<Type> argTypes;
            for (auto arg : args) {
                argTypes.push_back(arg.getType());
            }
            auto funcType = builder.getFunctionType(argTypes, builder.getI8Type().getPointerTo());
            funcOp = func::FuncOp::create(loc, funcName, funcType);
            module.push_back(funcOp);
        }
        
        // Create call
        auto callOp = builder.create<func::CallOp>(loc, funcOp, std::vector<Value>(args));
        return callOp.getResults();
    }
};

struct ExecutionContext {
    static CallableWrapper allocStateRaw(OpBuilder& builder, Location loc) {
        return CallableWrapper("exec_alloc_state_raw", builder, loc);
    }
};

} // namespace test_runtime

TEST_F(RuntimeWrapperTest, UnderstandCallPattern) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function to test in
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a size value
    auto sizeVal = builder.create<arith::ConstantIndexOp>(loc, 64);
    
    // This mimics the LingoDB pattern:
    // rt::ExecutionContext::allocStateRaw(rewriter, loc)({bytes})[0]
    auto result = test_runtime::ExecutionContext::allocStateRaw(builder, loc)({sizeVal})[0];
    
    EXPECT_TRUE(result);
    EXPECT_TRUE(result.getType().isa<LLVM::LLVMPointerType>());
}

TEST_F(RuntimeWrapperTest, CallWithMultipleArgs) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // The pattern also supports multiple arguments
    auto val1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    auto val2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Test a hypothetical setTupleCount pattern
    struct TestRuntime {
        static CallableWrapper setTupleCount(OpBuilder& builder, Location loc) {
            return CallableWrapper("exec_set_tuple_count", builder, loc);
        }
    };
    
    // This would be: rt::ExecutionContext::setTupleCount(rewriter, loc)({id, count})
    TestRuntime::setTupleCount(builder, loc)({val1, val2});
    
    // Verify the call was created
    module.walk([&](func::CallOp callOp) {
        EXPECT_EQ(callOp.getCallee(), "exec_set_tuple_count");
        EXPECT_EQ(callOp.getNumOperands(), 2);
    });
}