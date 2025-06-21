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

// Global context for tuple scanning - used by external function
struct TupleScanContext {
    TableScanDesc scanDesc;
    TupleDesc tupdesc;
    bool hasMore;
    int64_t currentValue;
};

static TupleScanContext* g_scan_context = nullptr;

extern "C" int64_t get_next_tuple() {
    if (!g_scan_context) {
        return -1;
    }
    
    HeapTuple tuple = heap_getnext(g_scan_context->scanDesc, ForwardScanDirection);
    if (tuple == NULL) {
        g_scan_context->hasMore = false;
        return -2;
    }
    
    bool isNull;
    Datum value = heap_getattr(tuple, 1, g_scan_context->tupdesc, &isNull);
    
    if (isNull) {
        return -3;
    }
    
    int64_t intValue = DatumGetInt64(value);
    g_scan_context->currentValue = intValue;
    g_scan_context->hasMore = true;
    
    return intValue;
}

auto run_mlir(int64_t intValue) -> void {
    PostgreSQLLogger logger;
    mlir_runner::run_mlir_core(intValue, logger);
}

auto run_mlir_with_tuple_scan(TableScanDesc scanDesc, TupleDesc tupdesc) -> void {
    PostgreSQLLogger logger;
    
    TupleScanContext scanContext = {scanDesc, tupdesc, true, 0};
    g_scan_context = &scanContext;
    
    mlir_runner::ExternalFunction tupleReader = []() -> int64_t {
        return get_next_tuple();
    };
    
    mlir_runner::run_mlir_with_external_func(0, tupleReader, logger);
    
    g_scan_context = nullptr;
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

    // Use MLIR JIT with external function call to read tuples
    // This demonstrates calling PostgreSQL's heap_getnext from within MLIR JIT
    run_mlir_with_tuple_scan(scanDesc, tupdesc);

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return true;
}
