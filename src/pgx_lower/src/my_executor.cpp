#include "my_executor.h"

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
    bool value1 = false;
    // Create MLIR context and builder
    mlir::MLIRContext context;
    elog(NOTICE, "MLIRContext symbol address: %p", (void *)&context);
    // Register required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    // Register the conversion pipeline
    registerConversionPipeline();

    std::unique_ptr<mlir::ExecutionEngine> engine;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    // Create a simple MLIR program that prints this number
    mlir::Location loc = builder.getUnknownLoc();

    // Create a function that returns i64
    auto funcType = builder.getFunctionType({}, {builder.getI64Type()});
    auto func = mlir::func::FuncOp::create(loc, "main", funcType);
    func.setPrivate();
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    auto constOp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(intValue));
    // Return the constant value
    builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
    module.push_back(func);

    // Verify the module
    if (mlir::failed(mlir::verify(module))) {
        elog(ERROR, "MLIR module verification failed");
        return;
    }

    // Print the MLIR program
    elog(NOTICE, "Generated MLIR program:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    elog(NOTICE, "MLIR: %s", mlirStr.c_str());

    // Lower to LLVM dialect
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    if (mlir::failed(pm.run(module))) {
        elog(ERROR, "Failed to lower MLIR module to LLVM dialect");
        return;
    }
    elog(NOTICE, "Lowered MLIR to LLVM dialect!");

    // JIT execute
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        elog(ERROR, "Failed to create MLIR ExecutionEngine");
        return;
    }
    elog(NOTICE, "Created MLIR ExecutionEngine!");
    engine = std::move(*maybeEngine);

    int64_t result = 0;
    llvm::Error err = engine->invoke("main", &result);
    elog(NOTICE, "Invoked MLIR JIT-compiled function!");
    if (err) {
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        os << err;
        elog(ERROR, "Failed to invoke MLIR JIT-compiled function: %s",
             errMsg.c_str());
        llvm::consumeError(std::move(err));
        return;
    }
    elog(NOTICE, "MLIR JIT returned: %ld", result);
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
