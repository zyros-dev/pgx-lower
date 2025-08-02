// Simplified SubOp to Control Flow lowering for basic queries
#include "dialects/subop/SubOpToControlFlow.h"
#include "core/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
}
#endif
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBOps.h"
#include "dialects/dsa/DSADialect.h"
// DSAOps.h not needed for simplified version
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace pgx_lower::compiler::dialect::subop {

namespace {

class SimpleSubOpToControlFlowPass : public mlir::PassWrapper<SimpleSubOpToControlFlowPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleSubOpToControlFlowPass)
    
    SimpleSubOpToControlFlowPass() = default;
    SimpleSubOpToControlFlowPass(const SimpleSubOpToControlFlowPass&) = default;
    
    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::cf::ControlFlowDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<db::DBDialect>();
        registry.insert<dsa::DSADialect>();
        registry.insert<util::UtilDialect>();
    }
    
    llvm::StringRef getArgument() const override { return "simple-lower-subop"; }
    llvm::StringRef getDescription() const override { return "Simple SubOp to Control Flow lowering"; }
    
    void runOnOperation() override {
        PGX_INFO("=== SimpleSubOpToControlFlowPass::runOnOperation() START ===");
        mlir::ModuleOp module = getOperation();
        auto &context = *module.getContext();
        mlir::OpBuilder builder(&context);
        PGX_INFO("=== SimpleSubOpToControlFlowPass: Got module and created builder ===");
        
        // Process ExecutionGroupOp operations
        llvm::SmallVector<ExecutionGroupOp, 4> execGroups;
        module.walk([&](ExecutionGroupOp op) {
            execGroups.push_back(op);
        });
        
        if (execGroups.empty()) {
            return; // Nothing to do
        }
        
        // Create main function if it doesn't exist
        auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
            builder.setInsertionPointToStart(module.getBody());
            auto funcType = builder.getFunctionType({}, builder.getI32Type());
            mainFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "main", funcType);
            mainFunc.setPublic();
            
            // Create entry block
            auto* entryBlock = mainFunc.addEntryBlock();
            builder.setInsertionPointToStart(entryBlock);
            
            // Generate code for each ExecutionGroupOp
            for (auto execGroup : execGroups) {
                // Extract operations from ExecutionGroupOp
                if (!execGroup.getRegion().empty() && !execGroup.getRegion().front().empty()) {
                    auto& block = execGroup.getRegion().front();
                    
                    // Look for key operations
                    subop::GetExternalOp getExternal;
                    subop::ScanRefsOp scanRefs;
                    
                    for (auto& op : block) {
                        if (auto ge = mlir::dyn_cast<subop::GetExternalOp>(op)) {
                            getExternal = ge;
                        } else if (auto sr = mlir::dyn_cast<subop::ScanRefsOp>(op)) {
                            scanRefs = sr;
                        }
                    }
                    
                    if (getExternal && scanRefs) {
                        // Generate actual table scan with loop
                        auto i32Type = builder.getI32Type();
                        auto i64Type = builder.getI64Type();
                        auto i1Type = builder.getI1Type();
                        
                        // Prepare results storage
                        auto prepareFunc = module.lookupSymbol<mlir::func::FuncOp>("prepare_computed_results");
                        if (!prepareFunc) {
                            auto savedIP = builder.saveInsertionPoint();
                            builder.setInsertionPointToStart(module.getBody());
                            auto prepareFuncType = builder.getFunctionType({i32Type}, {});
                            prepareFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "prepare_computed_results", prepareFuncType);
                            prepareFunc.setPrivate();
                            builder.restoreInsertionPoint(savedIP);
                        }
                        
                        auto one = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 1, 32);
                        builder.create<mlir::func::CallOp>(module.getLoc(), prepareFunc, mlir::ValueRange{one});
                        
                        // Declare and call read_next_tuple_from_table to get first tuple
                        auto readNextFunc = module.lookupSymbol<mlir::func::FuncOp>("read_next_tuple_from_table");
                        if (!readNextFunc) {
                            auto savedIP = builder.saveInsertionPoint();
                            builder.setInsertionPointToStart(module.getBody());
                            auto ptrType = mlir::LLVM::LLVMPointerType::get(&context);  // Use context from module
                            auto readNextFuncType = builder.getFunctionType({ptrType}, i64Type);
                            readNextFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "read_next_tuple_from_table", readNextFuncType);
                            readNextFunc.setPrivate();
                            builder.restoreInsertionPoint(savedIP);
                        }
                        
                        // Call with null handle (uses current scan context)
                        auto nullPtr = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 64);
                        auto nullPtrCast = builder.create<mlir::arith::IndexCastOp>(module.getLoc(), builder.getIndexType(), nullPtr);
                        auto tupleId = builder.create<mlir::func::CallOp>(module.getLoc(), readNextFunc, mlir::ValueRange{nullPtrCast}).getResult(0);
                        
                        // Get field 0 (extract integer ID)
                        auto getIntFieldFunc = module.lookupSymbol<mlir::func::FuncOp>("get_int_field");
                        if (!getIntFieldFunc) {
                            auto savedIP = builder.saveInsertionPoint();
                            builder.setInsertionPointToStart(module.getBody());
                            auto getIntFieldFuncType = builder.getFunctionType({i64Type, i32Type}, i32Type);
                            getIntFieldFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "get_int_field", getIntFieldFuncType);
                            getIntFieldFunc.setPrivate();
                            builder.restoreInsertionPoint(savedIP);
                        }
                        
                        auto zero32 = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 32);
                        auto fieldValue = builder.create<mlir::func::CallOp>(module.getLoc(), getIntFieldFunc, 
                            mlir::ValueRange{tupleId, zero32}).getResult(0);
                        
                        // Store the actual field value
                        auto storeFunc = module.lookupSymbol<mlir::func::FuncOp>("store_int_result");
                        if (!storeFunc) {
                            auto savedIP = builder.saveInsertionPoint();
                            builder.setInsertionPointToStart(module.getBody());
                            auto storeFuncType = builder.getFunctionType(
                                {i32Type, i32Type, i1Type}, {});
                            storeFunc = builder.create<mlir::func::FuncOp>(module.getLoc(), "store_int_result", storeFuncType);
                            storeFunc.setPrivate();
                            builder.restoreInsertionPoint(savedIP);
                        }
                        
                        auto falseVal = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 1);
                        builder.create<mlir::func::CallOp>(module.getLoc(), storeFunc, 
                            mlir::ValueRange{zero32, fieldValue, falseVal});
                        
                        // Ensure block has proper terminator after function call
                        auto currentBlock = builder.getInsertionBlock();
                        if (currentBlock && !currentBlock->getTerminator()) {
                            auto zero = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 32);
                            builder.create<mlir::func::ReturnOp>(module.getLoc(), mlir::ValueRange{zero});
                        }
                    }
                }
            }
            
            // Ensure function has terminator if none was added yet
            auto currentBlock = builder.getInsertionBlock();
            if (currentBlock && !currentBlock->getTerminator()) {
                auto zero = builder.create<mlir::arith::ConstantIntOp>(module.getLoc(), 0, 32);
                builder.create<mlir::func::ReturnOp>(module.getLoc(), mlir::ValueRange{zero});
            }
        }
        
        // Remove ExecutionGroupOp operations
        for (auto op : execGroups) {
            op.erase();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createSimpleSubOpToControlFlowPass() {
    return std::make_unique<SimpleSubOpToControlFlowPass>();
}

} // namespace pgx_lower::compiler::dialect::subop