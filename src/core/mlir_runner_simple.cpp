#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/LowerSubOpToDB.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "core/mlir_runner.h"
#include "logging/mlir_logger_pg.h"

using namespace mlir;

namespace pgx_lower::compiler {

bool MLIRRunner::runLoweringPipeline(mlir::ModuleOp module) {
    MLIRLoggerPG& logger = MLIRLoggerPG::getInstance();
    MLIRContext& context = *module.getContext();
    
    logger.notice("=== Running FULL LingoDB lowering pipeline ===");
    logger.notice("Starting LingoDB lowering pipeline...");
    
    // Phase 1: RelAlg → SubOp lowering
    logger.notice("Phase 1: RelAlg → SubOp lowering");
    {
        auto pm1 = mlir::PassManager(&context);
        pm1.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
        if (failed(pm1.run(module))) {
            logger.error("Phase 1 (RelAlg → SubOp) failed!");
            module.dump();
            return false;
        }
    }
    logger.notice("Phase 1 completed successfully");
    
    // Phase 2: Complete SubOp transform pipeline using LingoDB's proven approach
    logger.notice("Phase 2: Running complete SubOp transform pipeline");
    {
        auto pm2 = mlir::PassManager(&context);
        // Use LingoDB's complete pipeline instead of individual passes
        pgx_lower::compiler::dialect::subop::createLowerSubOpPipeline(pm2);
        
        logger.notice("Running complete LingoDB SubOp pipeline...");
        if (failed(pm2.run(module))) {
            logger.error("LingoDB SubOp transform pipeline failed!");
            module.dump();
            return false;
        }
        logger.notice("LingoDB SubOp transform pipeline completed successfully!");
        
        // Dump module after complete SubOp transformation
        logger.notice("Module after complete SubOp transformation:");
        module.dump();
    }
    
    // Continue with remaining passes in pm
    logger.notice("Adding remaining passes to main pipeline...");
    auto pm = mlir::PassManager(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    
    if (failed(pm.run(module))) {
        logger.error("Final canonicalization passes failed!");
        return false;
    }
    
    logger.notice("=== LingoDB lowering pipeline completed successfully! ===");
    return true;
}

} // namespace pgx_lower::compiler