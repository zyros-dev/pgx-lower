#include "lingodb/mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "lingodb/runtime/Database.h"
#include "pgx-lower/utility/logging.h"

std::shared_ptr<runtime::Database> staticDB = {};
void mlir::relalg::setStaticDB(std::shared_ptr<runtime::Database> db) {
   PGX_WARNING("Warning: setting static database, should only be used in combination with mlir-db-opt");
   staticDB = db;
}
void mlir::relalg::createQueryOptPipeline(OpPassManager& pm/*, runtime::Database* db*/) {
   pm.addNestedPass<func::FuncOp>(createSimplifyAggregationsPass());
   pm.addNestedPass<func::FuncOp>(createExtractNestedOperatorsPass());
   pm.addPass(createCSEPass());
   pm.addPass(createCanonicalizerPass());
   pm.addNestedPass<func::FuncOp>(createDecomposeLambdasPass());
   pm.addPass(createCanonicalizerPass());
   pm.addNestedPass<func::FuncOp>(createImplicitToExplicitJoinsPass());
   pm.addNestedPass<func::FuncOp>(createPushdownPass());
   pm.addNestedPass<func::FuncOp>(createUnnestingPass());
   // if (db) {
   //    pm.addNestedPass<::mlir::func::FuncOp>(mlir::relalg::createAttachMetaDataPass(*db));
   // }
   pm.addNestedPass<func::FuncOp>(createReduceGroupByKeysPass());
   pm.addNestedPass<func::FuncOp>(createExpandTransitiveEqualities());
   pm.addNestedPass<func::FuncOp>(createOptimizeJoinOrderPass());
   // if (db) {
   //    pm.addNestedPass<::mlir::func::FuncOp>(mlir::relalg::createDetachMetaDataPass());
   // }
   pm.addNestedPass<func::FuncOp>(createCombinePredicatesPass());
   pm.addNestedPass<func::FuncOp>(createOptimizeImplementationsPass());
   pm.addNestedPass<func::FuncOp>(createIntroduceTmpPass());
   pm.addPass(createCanonicalizerPass());
}
void mlir::relalg::registerQueryOptimizationPasses() {
   registerPass([]() -> std::unique_ptr<Pass> {
      return createExtractNestedOperatorsPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createDecomposeLambdasPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createImplicitToExplicitJoinsPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createUnnestingPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createPushdownPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createOptimizeJoinOrderPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createCombinePredicatesPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createOptimizeImplementationsPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createIntroduceTmpPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createSimplifyAggregationsPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createReduceGroupByKeysPass();
   });
   registerPass([]() -> std::unique_ptr<Pass> {
      return createExpandTransitiveEqualities();
   });

   ::mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "relalg-query-opt",
      "",
      [](OpPassManager& pm) { return createQueryOptPipeline(pm/*, staticDB.get()*/); });
}
