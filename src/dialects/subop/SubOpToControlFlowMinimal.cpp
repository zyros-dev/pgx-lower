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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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
    }
    
    llvm::StringRef getArgument() const override { return "minimal-lower-subop"; }
    llvm::StringRef getDescription() const override { return "Minimal SubOp to Control Flow lowering"; }
    
    void runOnOperation() override {
#ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "=== MinimalSubOpToControlFlowPass::runOnOperation() START ===");
#endif
        mlir::ModuleOp module = getOperation();
        mlir::OpBuilder builder(&getContext());
        
        // Find ExecutionGroupOp operations to remove
        llvm::SmallVector<ExecutionGroupOp, 4> execGroups;
        module.walk([&](ExecutionGroupOp op) {
            execGroups.push_back(op);
        });
        
#ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "Found %d ExecutionGroupOp operations to remove", (int)execGroups.size());
#endif
        
        // Create main function if it doesn't exist
        auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
#ifdef POSTGRESQL_EXTENSION
            elog(NOTICE, "Creating main function - starting with basic version");
#endif
            builder.setInsertionPointToStart(module.getBody());
            auto funcType = builder.getFunctionType({}, builder.getI32Type());
            mainFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "main", funcType);
            mainFunc.setPublic();
            
            // Create entry block with basic PostgreSQL result preparation
            auto* entryBlock = mainFunc.addEntryBlock();
            builder.setInsertionPointToStart(entryBlock);
            
            // Add minimal PostgreSQL functionality - prepare results
            auto i32Type = builder.getI32Type();
            
            // Declare prepare_computed_results function
            auto prepareFunc = module.lookupSymbol<mlir::func::FuncOp>("prepare_computed_results");
            if (!prepareFunc) {
                auto savedIP = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(module.getBody());
                auto prepareFuncType = builder.getFunctionType({i32Type}, {});
                prepareFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "prepare_computed_results", prepareFuncType);
                prepareFunc.setPrivate();
                builder.restoreInsertionPoint(savedIP);
            }
            
            // Call prepare_computed_results(1) - prepare for 1 column
            auto one = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 1, 32);
            builder.create<mlir::func::CallOp>(module.getLoc(), prepareFunc, mlir::ValueRange(one));
            
            auto zero = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 32);
            builder.create<mlir::func::ReturnOp>(module.getLoc(), mlir::ValueRange(zero));
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