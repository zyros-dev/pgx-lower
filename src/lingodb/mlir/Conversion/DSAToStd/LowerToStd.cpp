// Lightweight version with minimal MLIR includes
#include "DSAToStdPatterns.h"
#include "pgx-lower/execution/logging.h"

// PostgreSQL headers for exception handling
#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}
#endif

// Only essential includes for the actual conversion logic
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "runtime-defs/DataSourceIteration.h"

using namespace mlir;

namespace {

class ScanSourceLowering : public OpConversionPattern<mlir::dsa::ScanSource> {
   public:
   using OpConversionPattern<mlir::dsa::ScanSource>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::ScanSource op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MLIR_PGX_DEBUG("DSA", "ScanSourceLowering: ENTRY");
      MLIR_PGX_DEBUG("DSA", "ScanSourceLowering: Original ScanSource result type check...");
      auto originalType = op.getResult().getType();
      if (auto genIterType = originalType.dyn_cast_or_null<mlir::dsa::GenericIterableType>()) {
         MLIR_PGX_DEBUG("DSA", "ScanSourceLowering: Original type is GenericIterableType with name: " + genIterType.getIteratorName());
      }
      
      std::vector<Type> types;
      auto parentModule = op->getParentOfType<ModuleOp>();
      ::mlir::func::FuncOp funcOp = parentModule.lookupSymbol<::mlir::func::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         ::mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto funcType = rewriter.getFunctionType({}, {mlir::util::RefType::get(getContext(), rewriter.getI8Type())});
         funcOp = rewriter.create<::mlir::func::FuncOp>(op->getLoc(), "rt_get_execution_context", funcType);
         // Fix visibility issue - external functions must be private
         funcOp.setVisibility(::mlir::func::FuncOp::Visibility::Private);
      }

      ::mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, ::mlir::ValueRange{}).getResult(0);
      ::mlir::Value description = rewriter.create<mlir::util::CreateConstVarLen>(op->getLoc(), mlir::util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());
      
      // CRITICAL: Ensure rewriter has a valid insertion block before calling runtime functions
      if (!rewriter.getInsertionBlock()) {
         MLIR_PGX_ERROR("DSA", "ScanSourceLowering: rewriter has no insertion block!");
         return failure();
      }
      
      MLIR_PGX_DEBUG("DSA", "Calling rt::DataSourceIteration::start");
      auto rawPtr = rt::DataSourceIteration::start(rewriter, op->getLoc())({executionContext, description})[0];
      MLIR_PGX_DEBUG("DSA", "Successfully called rt::DataSourceIteration::start");
      
      MLIR_PGX_DEBUG("DSA", "ScanSourceLowering: Replacing op with i8* pointer");
      rewriter.replaceOp(op, rawPtr);
      return success();
   }
};

// Function declared in DSAToStdPatterns.h, implemented in PatternRegistry.cpp

} // end anonymous namespace

namespace {
struct DSAToStdLoweringPass : public PassWrapper<DSAToStdLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "lower-dsa"; }

   DSAToStdLoweringPass() {
      PGX_INFO("DSAToStdLoweringPass: Constructor ENTRY");
   }
   
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::dsa::DSADialect, scf::SCFDialect, 
                     mlir::cf::ControlFlowDialect, util::UtilDialect, 
                     memref::MemRefDialect, arith::ArithDialect>();
   }
   
   void runOnOperation() final {
      auto module = getOperation();
      getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);
      
      ConversionTarget target(getContext());
      target.addLegalOp<ModuleOp>();
      target.addLegalOp<UnrealizedConversionCastOp>();
      
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<scf::SCFDialect>();
      target.addLegalDialect<cf::ControlFlowDialect>();
      target.addLegalDialect<util::UtilDialect>();
      
      target.addIllegalDialect<mlir::dsa::DSADialect>();
      
      TypeConverter typeConverter;
      typeConverter.addConversion([&](mlir::Type type) { return type; });

      // Type conversions
      typeConverter.addConversion([&](mlir::dsa::TableType tableType) {
         return mlir::util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
      });
      typeConverter.addConversion([&](mlir::dsa::TableBuilderType tableType) {
         return mlir::util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
      });

      RewritePatternSet patterns(&getContext());
      
      // Register lightweight patterns
      patterns.insert<ScanSourceLowering>(typeConverter, &getContext());
      
      // Register heavy patterns from separate compilation unit
      mlir::dsa::registerAllDSAToStdPatterns(typeConverter, patterns, target);

      // Wrap MLIR conversion in PostgreSQL exception handling to prevent memory corruption
      bool conversionSucceeded = false;
      #ifdef POSTGRESQL_EXTENSION
      PG_TRY();
      {
      #endif
          if (failed(applyFullConversion(module, target, std::move(patterns)))) {
             PGX_ERROR("[DSAToStd] applyFullConversion FAILED");
             conversionSucceeded = false;
          } else {
             PGX_INFO("[DSAToStd] applyFullConversion SUCCEEDED");  
             conversionSucceeded = true;
          }
      #ifdef POSTGRESQL_EXTENSION
      }
      PG_CATCH();
      {
          PGX_ERROR("[DSAToStd] PostgreSQL exception caught during applyFullConversion");
          PGX_ERROR("[DSAToStd] This indicates memory corruption or signal handling conflict");
          conversionSucceeded = false;
          // Re-throw to let PostgreSQL handle the cleanup
          PG_RE_THROW();
      }
      PG_END_TRY();
      #endif
      
      if (!conversionSucceeded) {
          signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<::mlir::Pass> mlir::dsa::createLowerToStdPass() {
   PGX_INFO("DSAToStd: ENTRY createLowerToStdPass");
   auto pass = std::make_unique<DSAToStdLoweringPass>();
   PGX_INFO("DSAToStd: AFTER DSAToStdLoweringPass constructor - success");
   return pass;
}