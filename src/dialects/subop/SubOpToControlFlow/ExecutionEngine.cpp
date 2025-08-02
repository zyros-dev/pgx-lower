#include "dialects/subop/SubOpToControlFlow.h"
#include "core/logging.h"
#include "Headers/SubOpToControlFlowUtilities.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
}
#endif

#include "dialects/util/UtilToLLVMPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/util/FunctionHelper.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "compiler/runtime/helpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <stack>
#include <unordered_set>
using namespace mlir;


namespace {
using namespace pgx_lower::compiler::dialect;
namespace rt = pgx_lower::compiler::runtime;
using pgx_lower::compiler::dialect::subop_to_cf::EntryStorageHelper;

// EntryStorageHelper is now defined in SubOpToControlFlowUtilities.h

// Type conversion utility functions
static mlir::TupleType convertTuple(mlir::TupleType tupleType, mlir::TypeConverter& typeConverter) {
    std::vector<mlir::Type> convertedTypes;
    for (auto type : tupleType.getTypes()) {
        convertedTypes.push_back(typeConverter.convertType(type));
    }
    return mlir::TupleType::get(tupleType.getContext(), convertedTypes);
}

// Pass Declaration
struct SubOpToControlFlowLoweringPass
   : public mlir::PassWrapper<SubOpToControlFlowLoweringPass, OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubOpToControlFlowLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-subop-to-cf"; }

   SubOpToControlFlowLoweringPass() {
      PGX_INFO("=== SubOpToControlFlowLoweringPass constructor called ===");
   }
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};

// Simplified core execution step handling function
void handleExecutionStepCPU(subop::ExecutionStepOp step, subop::ExecutionGroupOp executionGroup, mlir::IRMapping& mapping, mlir::TypeConverter& typeConverter) {
   auto* ctxt = step->getContext();
   
   PGX_DEBUG("Handling ExecutionStepOp with simplified CPU implementation");
   
   // For now, just map inputs to arguments directly without complex pattern rewriting
   for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
      mlir::Value input = mapping.lookup(param);
      if (!mlir::cast<mlir::BoolAttr>(isThreadLocal).getValue()) {
         mapping.map(arg, input);
      } else {
         // Handle thread local case with simplified mapping
         mapping.map(arg, input);
      }
   }
   
   // Handle return operation
   auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
   for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
      mapping.map(o, mapping.lookup(i));
   }
}

} // namespace

// Main pass implementation
void SubOpToControlFlowLoweringPass::runOnOperation() {
   PGX_INFO("=== SubOpToControlFlowLoweringPass::runOnOperation() START ===");
   MLIR_PGX_DEBUG("SubOp", "SubOpToControlFlowLoweringPass is executing!");
   
   try {
      auto module = getOperation();
      
      PGX_INFO("Got module operation successfully");
   
   // Load UtilDialect if not already loaded
   auto* utilDialect = getContext().getLoadedDialect<util::UtilDialect>();
   if (!utilDialect) {
      PGX_INFO("UtilDialect not loaded, loading now...");
      utilDialect = getContext().getOrLoadDialect<util::UtilDialect>();
      if (!utilDialect) {
         PGX_ERROR("Failed to load UtilDialect!");
         signalPassFailure();
         return;
      }
   }
   
   PGX_INFO("UtilDialect loaded successfully");
   utilDialect->getFunctionHelper().setParentModule(module);
   auto* ctxt = &getContext();
   
   // Check if main function exists
   auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
   mlir::Block* mainBlock = nullptr;
   if (!mainFunc) {
      MLIR_PGX_DEBUG("SubOp", "Creating main function");
      // Create main function if it doesn't exist
      mlir::OpBuilder builder(module);
      builder.setInsertionPointToEnd(module.getBody());
      
      auto mainFuncType = builder.getFunctionType({}, builder.getI32Type());
      mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFuncType);
      
      // Create entry block
      mainBlock = mainFunc.addEntryBlock();
   } else {
      MLIR_PGX_DEBUG("SubOp", "Main function already exists");
      mainBlock = &mainFunc.getBody().front();
   }

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::Type t) { return t; });
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](subop::TableType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::LocalTableType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ResultTableType t) -> Type {
      std::vector<mlir::Type> types;
      for (size_t i = 0; i < t.getMembers().getTypes().size(); i++) {
         types.push_back(util::RefType::get(mlir::IntegerType::get(t.getContext(), 8)));
      }
      return util::RefType::get(mlir::TupleType::get(ctxt, types));
   });
   typeConverter.addConversion([&](subop::BufferType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::SortedViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getBasedOn().getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ArrayType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ContinuousViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getBasedOn().getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::SimpleStateType t) -> Type {
      return util::RefType::get(t.getContext(), EntryStorageHelper(nullptr, t.getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::HashMapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::PreAggrHtFragmentType t) -> Type {
      return util::RefType::get(t.getContext(), util::RefType::get(t.getContext(), subop_to_control_flow::getHtEntryType(t, typeConverter)));
   });
   typeConverter.addConversion([&](subop::PreAggrHtType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HashMultiMapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ThreadLocalType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::SegmentTreeViewType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HashIndexedViewType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HeapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ExternalHashIndexType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ListType t) -> Type {
      if (auto lookupEntryRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(t.getT())) {
         if (mlir::isa<subop::HashMapType>(lookupEntryRefType.getState())) {
            return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
         }
         if (auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupEntryRefType.getState())) {
            return util::RefType::get(t.getContext(), subop_to_control_flow::getHashMultiMapEntryType(hashMultiMapType, typeConverter));
         }
         if (auto externalHashIndexRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexType>(t.getT())) {
            return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
         }
         return mlir::TupleType::get(t.getContext(), {util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8)), mlir::IndexType::get(t.getContext())});
      }
      if (auto hashMapEntryRefType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMapEntryRefType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMultiMapEntryRefType = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), subop_to_control_flow::getHashMultiMapEntryType(hashMultiMapEntryRefType.getHashMultimap(), typeConverter));
      }
      if (auto externalHashIndexRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      return mlir::Type();
   });
   typeConverter.addConversion([&](subop::HashMapEntryRefType t) -> Type {
      auto hashMapType = t.getHashMap();
      return util::RefType::get(t.getContext(), subop_to_control_flow::getHtKVType(hashMapType, typeConverter));
   });
   typeConverter.addConversion([&](subop::PreAggrHTEntryRefType t) -> Type {
      auto hashMapType = t.getHashMap();
      return util::RefType::get(t.getContext(), subop_to_control_flow::getHtKVType(hashMapType, typeConverter));
   });
   typeConverter.addConversion([&](subop::LookupEntryRefType t) -> Type {
      if (mlir::isa<subop::HashMapType, subop::PreAggrHtFragmentType>(t.getState())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(t.getState())) {
         return util::RefType::get(t.getContext(), subop_to_control_flow::getHashMultiMapEntryType(hashMultiMapType, typeConverter));
      }
      return mlir::TupleType::get(t.getContext(), {util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8)), mlir::IndexType::get(t.getContext())});
   });

   // Walk over ExecutionGroupOps (queries)
   std::vector<mlir::Operation*> toRemove;
   module->walk([&, mainFunc, mainBlock](subop::ExecutionGroupOp executionGroup) {
      PGX_INFO("=== Processing ExecutionGroupOp in walk ===");
      mlir::IRMapping mapping;
      
      // Check if we have any ExecutionStepOp operations
      bool hasExecutionSteps = false;
      for (auto& op : executionGroup.getRegion().front().getOperations()) {
         if (mlir::isa<subop::ExecutionStepOp>(op)) {
            hasExecutionSteps = true;
            break;
         }
      }
      PGX_INFO("=== ExecutionGroupOp has ExecutionSteps: " + std::string(hasExecutionSteps ? "true" : "false") + " ===");
      
      if (!hasExecutionSteps) {
         // Handle simple case without ExecutionSteps - generate PostgreSQL code directly
         MLIR_PGX_DEBUG("SubOp", "Handling simple ExecutionGroupOp without ExecutionSteps");
         PGX_INFO("=== Handling simple case without ExecutionSteps ===");
         
         mlir::OpBuilder mainBuilder(mainFunc);
         mainBuilder.setInsertionPointToEnd(mainBlock);
         
         // Pre-add terminator to prevent MLIR validation errors
         if (!mainBlock->getTerminator()) {
            auto preZero = mainBuilder.create<mlir::arith::ConstantIntOp>(mainBuilder.getUnknownLoc(), 0, 32);
            auto preTerminator = mainBuilder.create<mlir::func::ReturnOp>(mainBuilder.getUnknownLoc(), mlir::ValueRange{preZero});
            MLIR_PGX_INFO("SubOp", "Pre-added terminator to prevent MLIR validation errors");
            mainBuilder.setInsertionPoint(preTerminator);
         }
         
         // Process each operation in the ExecutionGroupOp
         for (auto& op : executionGroup.getRegion().front()) {
            if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) {
               continue;
            }
            
            MLIR_PGX_DEBUG("SubOp", "Processing operation: " + op.getName().getStringRef().str());
            
            // Handle specific SubOp operations - Generate PostgreSQL code
            if (auto getExternal = mlir::dyn_cast<subop::GetExternalOp>(op)) {
               MLIR_PGX_DEBUG("SubOp", "Generating PostgreSQL code for GetExternalOp");
               // Generate table access preparation code
               auto i32Type = mainBuilder.getI32Type();
               auto one = mainBuilder.create<mlir::arith::ConstantIntOp>(mainBuilder.getUnknownLoc(), 1, 32);
               
               // Prepare results storage
               auto prepareFunc = module.lookupSymbol<mlir::func::FuncOp>("prepare_computed_results");
               if (!prepareFunc) {
                  auto savedIP = mainBuilder.saveInsertionPoint();
                  mainBuilder.setInsertionPointToStart(module.getBody());
                  auto prepareFuncType = mainBuilder.getFunctionType({i32Type}, {});
                  prepareFunc = mainBuilder.create<mlir::func::FuncOp>(mainBuilder.getUnknownLoc(), "prepare_computed_results", prepareFuncType);
                  prepareFunc.setPrivate();
                  mainBuilder.restoreInsertionPoint(savedIP);
               }
               auto prepareCallOp = mainBuilder.create<mlir::func::CallOp>(mainBuilder.getUnknownLoc(), prepareFunc, mlir::ValueRange{one});
               
               mapping.map(getExternal.getResult(), getExternal.getResult());
            } else if (auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(op)) {
               MLIR_PGX_DEBUG("SubOp", "Generating PostgreSQL code for ScanRefsOp");
               // Generate actual table scan with PostgreSQL runtime calls
               auto i32Type = mainBuilder.getI32Type();
               auto i64Type = mainBuilder.getI64Type();
               auto i1Type = mainBuilder.getI1Type();
               
               // Declare and call read_next_tuple_from_table to get first tuple
               auto readNextFunc = module.lookupSymbol<mlir::func::FuncOp>("read_next_tuple_from_table");
               if (!readNextFunc) {
                  auto savedIP = mainBuilder.saveInsertionPoint();
                  mainBuilder.setInsertionPointToStart(module.getBody());
                  auto ptrType = mlir::LLVM::LLVMPointerType::get(&getContext());
                  auto readNextFuncType = mainBuilder.getFunctionType({ptrType}, i64Type);
                  readNextFunc = mainBuilder.create<mlir::func::FuncOp>(mainBuilder.getUnknownLoc(), "read_next_tuple_from_table", readNextFuncType);
                  readNextFunc.setPrivate();
                  mainBuilder.restoreInsertionPoint(savedIP);
               }
               
               // Call with null handle (uses current scan context)
               auto nullPtr = mainBuilder.create<mlir::arith::ConstantIntOp>(mainBuilder.getUnknownLoc(), 0, 64);
               auto nullPtrCast = mainBuilder.create<mlir::arith::IndexCastOp>(mainBuilder.getUnknownLoc(), mainBuilder.getIndexType(), nullPtr);
               auto tupleId = mainBuilder.create<mlir::func::CallOp>(mainBuilder.getUnknownLoc(), readNextFunc, mlir::ValueRange{nullPtrCast}).getResult(0);
               
               // Get field 0 (extract integer ID)
               auto getIntFieldFunc = module.lookupSymbol<mlir::func::FuncOp>("get_int_field");
               if (!getIntFieldFunc) {
                  auto savedIP = mainBuilder.saveInsertionPoint();
                  mainBuilder.setInsertionPointToStart(module.getBody());
                  auto getIntFieldFuncType = mainBuilder.getFunctionType({i64Type, i32Type}, i32Type);
                  getIntFieldFunc = mainBuilder.create<mlir::func::FuncOp>(mainBuilder.getUnknownLoc(), "get_int_field", getIntFieldFuncType);
                  getIntFieldFunc.setPrivate();
                  mainBuilder.restoreInsertionPoint(savedIP);
               }
               
               auto zero32 = mainBuilder.create<mlir::arith::ConstantIntOp>(mainBuilder.getUnknownLoc(), 0, 32);
               auto fieldValue = mainBuilder.create<mlir::func::CallOp>(mainBuilder.getUnknownLoc(), getIntFieldFunc, 
                   mlir::ValueRange{tupleId, zero32}).getResult(0);
               
               // Store the actual field value
               auto storeFunc = module.lookupSymbol<mlir::func::FuncOp>("store_int_result");
               if (!storeFunc) {
                  auto savedIP = mainBuilder.saveInsertionPoint();
                  mainBuilder.setInsertionPointToStart(module.getBody());
                  auto storeFuncType = mainBuilder.getFunctionType(
                      {i32Type, i32Type, i1Type}, {});
                  storeFunc = mainBuilder.create<mlir::func::FuncOp>(mainBuilder.getUnknownLoc(), "store_int_result", storeFuncType);
                  storeFunc.setPrivate();
                  mainBuilder.restoreInsertionPoint(savedIP);
               }
               
               auto falseVal = mainBuilder.create<mlir::arith::ConstantIntOp>(mainBuilder.getUnknownLoc(), 0, 1);
               auto callOp = mainBuilder.create<mlir::func::CallOp>(mainBuilder.getUnknownLoc(), storeFunc, 
                   mlir::ValueRange{zero32, fieldValue, falseVal});

               mapping.map(scanRefs.getResult(), scanRefs.getResult());
            } else {
               // Map all results to themselves for now
               for (auto result : op.getResults()) {
                  mapping.map(result, result);
               }
            }
         }
      } else {
         // Handle ExecutionStepOp with simplified implementation
         for (auto& op : executionGroup.getRegion().front().getOperations()) {
            if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
               handleExecutionStepCPU(step, executionGroup, mapping, typeConverter);
            }
         }
      }
      auto returnOp = mlir::cast<subop::ExecutionGroupReturnOp>(executionGroup.getRegion().front().getTerminator());
      std::vector<mlir::Value> results;
      for (auto i : returnOp.getInputs()) {
         results.push_back(mapping.lookup(i));
      }
      executionGroup.replaceAllUsesWith(results);
      toRemove.push_back(executionGroup);
      PGX_INFO("=== ExecutionGroupOp processing completed successfully ===");
   });
   
   getOperation()->walk([&](subop::SetResultOp setResultOp) {
      mlir::OpBuilder builder(setResultOp);
      mlir::Value idVal = builder.create<mlir::arith::ConstantIntOp>(setResultOp.getLoc(), setResultOp.getResultId(), mlir::IntegerType::get(builder.getContext(), 32));
      // TODO Phase 5: Implement ExecutionContext::setResult MLIR wrapper
      // pgx_lower::compiler::runtime::ExecutionContext::setResult(builder, setResultOp->getLoc())({idVal, setResultOp.getState()});

      toRemove.push_back(setResultOp);
   });
   
   for (auto* op : toRemove) {
      PGX_INFO("=== About to erase operation: " + std::string(op->getName().getStringRef().data()) + " ===");
      op->dropAllReferences();
      op->dropAllDefinedValueUses();
      op->erase();
   }
   PGX_INFO("=== All operations erased successfully ===");
   
   std::vector<mlir::Operation*> defs;
   for (auto& op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&op)) {
         if (funcOp.getBody().empty()) {
            defs.push_back(&op);
         }
      }
   }
   for (auto* op : defs) {
      op->moveBefore(&module.getBody()->getOperations().front());
   }
   module->walk([&](tuples::GetParamVal getParamVal) {
      getParamVal.replaceAllUsesWith(getParamVal.getParam());
   });
   
   PGX_INFO("=== SubOpToControlFlowLoweringPass::runOnOperation() completing successfully ===");
   
   } catch (const std::exception& e) {
      PGX_ERROR("Exception in SubOpToControlFlowLoweringPass::runOnOperation(): " + std::string(e.what()));
      signalPassFailure();
      return;
   } catch (...) {
      PGX_ERROR("Unknown exception in SubOpToControlFlowLoweringPass::runOnOperation()");
      signalPassFailure();
      return;
   }
}

// Pipeline registration functions
std::unique_ptr<mlir::Pass> subop::createLowerSubOpPass() {
   PGX_INFO("=== createLowerSubOpPass() called ===");
   auto pass = std::make_unique<SubOpToControlFlowLoweringPass>();
   PGX_INFO("=== SubOpToControlFlowLoweringPass created successfully ===");
   return pass;
}

void subop::setCompressionEnabled(bool compressionEnabled) {
   EntryStorageHelper::compressionEnabled = compressionEnabled;
}

void subop::createLowerSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(subop::createGlobalOptPass());
   pm.addPass(subop::createFoldColumnsPass());
   pm.addPass(subop::createReuseLocalPass());
   pm.addPass(subop::createSpecializeSubOpPass(true));
   pm.addPass(subop::createNormalizeSubOpPass());
   pm.addPass(subop::createPullGatherUpPass());
   pm.addPass(subop::createParallelizePass());
   pm.addPass(subop::createEnforceOrderPass());
   pm.addPass(subop::createInlineNestedMapPass());
   pm.addPass(subop::createFinalizePass());
   pm.addPass(subop::createSplitIntoExecutionStepsPass());
   pm.addNestedPass<mlir::func::FuncOp>(subop::createParallelizePass());
   pm.addPass(subop::createSpecializeParallelPass());
   pm.addPass(subop::createPrepareLoweringPass());
   pm.addPass(subop::createLowerSubOpPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addPass(mlir::createCSEPass());
}

std::unique_ptr<mlir::Pass> subop::createLowerSubOpToControlFlowPass() {
   return subop::createLowerSubOpPass();
}

void subop::registerSubOpToControlFlowConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createLowerSubOpPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-subop",
      "",
      subop::createLowerSubOpPipeline);
}