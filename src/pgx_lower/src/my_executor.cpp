#include "my_executor.h"
#include "mlir_runner.h"
#include "mlir_logger.h"

#include "executor/executor.h"

extern "C" {
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "postgres.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
}

// Undefine PostgreSQL macros that conflict with LLVM
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

void registerConversionPipeline() {
    mlir::PassPipelineRegistration<>(
        "convert-to-llvm", "Convert MLIR to LLVM dialect",
        [](mlir::OpPassManager &pm) {
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createConvertSCFToCFPass());
        });
}

auto run_mlir(int64_t intValue) -> void {
    PostgreSQLLogger logger;
    mlir_runner::run_mlir_core(intValue, logger);
}

bool MyCppExecutor::execute(const QueryDesc *plan) {
    elog(NOTICE, "LLVM version: %d.%d.%d", LLVM_VERSION_MAJOR,
         LLVM_VERSION_MINOR, LLVM_VERSION_PATCH);
    if (!plan) {
        elog(ERROR, "QueryDesc is null");
        return false;
    }

    elog(NOTICE, "Inside C++ executor! Plan type: %d", plan->operation);
    elog(NOTICE, "Query text: %s",
         plan->sourceText ? plan->sourceText : "NULL");

    if (plan->operation != CMD_SELECT) {
        elog(NOTICE, "Not a SELECT statement, skipping");
        return false;
    }

    PlannedStmt *stmt = plan->plannedstmt;
    Plan *rootPlan = stmt->planTree;

    if (rootPlan->type != T_SeqScan) {
        elog(NOTICE,
             "Only simple table scans (SeqScan) are supported in raw mode.");
        return false;
    }

    SeqScan *scan = (SeqScan *)rootPlan;
    RangeTblEntry *rte =
        (RangeTblEntry *)list_nth(stmt->rtable, scan->scan.scanrelid - 1);
    Relation rel = table_open(rte->relid, AccessShareLock);

    TableScanDesc scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, NULL);
    TupleDesc tupdesc = RelationGetDescr(rel);
    HeapTuple tuple;

    // Create a simple MLIR program that prints the first number from the query
    if ((tuple = heap_getnext(scanDesc, ForwardScanDirection)) != NULL) {
        bool isNull;
        Datum value = heap_getattr(tuple, 1, tupdesc, &isNull);

        if (!isNull) {
            // Get the value as an integer
            int64_t intValue = DatumGetInt64(value);

            run_mlir(intValue);
        }
    }

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return true;
}
