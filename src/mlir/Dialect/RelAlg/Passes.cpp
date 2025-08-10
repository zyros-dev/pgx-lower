#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

std::shared_ptr<runtime::Database> staticDB = {};
void pgx::mlir::relalg::setStaticDB(std::shared_ptr<runtime::Database> db) {
   std::cerr << "Warning: setting static database, should only be used in combination with mlir-db-opt" << std::endl;
   staticDB = db;
}
void pgx::mlir::relalg::createQueryOptPipeline(mlir::OpPassManager& pm, runtime::Database* db) {
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createSimplifyAggregationsPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createExtractNestedOperatorsPass());
   pm.addPass(mlir::createCSEPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createDecomposeLambdasPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createImplicitToExplicitJoinsPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createPushdownPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createUnnestingPass());
   if (db) {
      pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createAttachMetaDataPass(*db));
   }
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createReduceGroupByKeysPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createExpandTransitiveEqualities());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createOptimizeJoinOrderPass());
   if (db) {
      pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createDetachMetaDataPass());
   }
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createCombinePredicatesPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createOptimizeImplementationsPass());
   pm.addNestedPass<mlir::func::FuncOp>(pgx::mlir::relalg::createIntroduceTmpPass());
   pm.addPass(mlir::createCanonicalizerPass());
}
void pgx::mlir::relalg::registerQueryOptimizationPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createExtractNestedOperatorsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createDecomposeLambdasPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createImplicitToExplicitJoinsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createUnnestingPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createPushdownPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createOptimizeJoinOrderPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createCombinePredicatesPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createOptimizeImplementationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createIntroduceTmpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createSimplifyAggregationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createReduceGroupByKeysPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return pgx::mlir::relalg::createExpandTransitiveEqualities();
   });

   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "relalg-query-opt",
      "",
      [](mlir::OpPassManager& pm) { return createQueryOptPipeline(pm, staticDB.get()); });
}
