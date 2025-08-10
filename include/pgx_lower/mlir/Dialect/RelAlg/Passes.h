#ifndef MLIR_DIALECT_RELALG_PASSES_H
#define MLIR_DIALECT_RELALG_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace runtime {
class Database;
} // end namespace runtime
namespace pgx::mlir::relalg {
std::unique_ptr<::mlir::Pass> createExtractNestedOperatorsPass();
std::unique_ptr<::mlir::Pass> createDecomposeLambdasPass();
std::unique_ptr<::mlir::Pass> createImplicitToExplicitJoinsPass();
std::unique_ptr<::mlir::Pass> createUnnestingPass();
std::unique_ptr<::mlir::Pass> createPushdownPass();
std::unique_ptr<::mlir::Pass> createOptimizeJoinOrderPass();
std::unique_ptr<::mlir::Pass> createCombinePredicatesPass();
std::unique_ptr<::mlir::Pass> createOptimizeImplementationsPass();
std::unique_ptr<::mlir::Pass> createIntroduceTmpPass();
std::unique_ptr<::mlir::Pass> createReduceGroupByKeysPass();
std::unique_ptr<::mlir::Pass> createExpandTransitiveEqualities();

std::unique_ptr<::mlir::Pass> createSimplifyAggregationsPass();
std::unique_ptr<::mlir::Pass> createAttachMetaDataPass(runtime::Database& db);
std::unique_ptr<::mlir::Pass> createDetachMetaDataPass();

void registerQueryOptimizationPasses();
void setStaticDB(std::shared_ptr<runtime::Database> db);
void createQueryOptPipeline(::mlir::OpPassManager& pm, runtime::Database* db);

} // namespace pgx::mlir::relalg

#endif // MLIR_DIALECT_RELALG_PASSES_H