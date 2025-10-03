#include "pgx-lower/execution/postgres/executor_c.h"
#include "pgx-lower/utility/logging_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "funcapi.h"
#include "postgres.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/guc_hooks.h"
#include "utils/tuplestore.h"

PG_MODULE_MAGIC;

extern bool g_extension_after_load;

static bool pgx_lower_log_enable = false;
static bool pgx_lower_log_debug = false;
static bool pgx_lower_log_ir = false;
static bool pgx_lower_log_io = false;
static bool pgx_lower_log_trace = false;
static char *pgx_lower_enabled_categories = NULL;

extern void pgx_update_log_settings(bool enable, bool debug, bool ir, bool io, bool trace, const char *categories);

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;

static bool try_cpp_executor_internal(const QueryDesc *queryDesc) {
    PGX_NOTICE_C("Calling C++ executor from C...");
    return try_cpp_executor_direct(queryDesc);
}

static void
custom_executor(QueryDesc *queryDesc, const ScanDirection direction, const uint64 count, const bool execute_once) {
    PGX_NOTICE_C("Custom executor is being executed in C!");

    const bool mlir_handled = try_cpp_executor_internal(queryDesc);

    if (!mlir_handled) {
        if (prev_ExecutorRun_hook) {
            prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
        }
        else {
            standard_ExecutorRun(queryDesc, direction, count, execute_once);
        }
    }
}

static void update_logging_bool(bool newval, void *extra) {
    pgx_update_log_settings(pgx_lower_log_enable, pgx_lower_log_debug, pgx_lower_log_ir, pgx_lower_log_io,
                            pgx_lower_log_trace, pgx_lower_enabled_categories);
}

static void update_logging_string(const char *newval, void *extra) {
    pgx_update_log_settings(pgx_lower_log_enable, pgx_lower_log_debug, pgx_lower_log_ir, pgx_lower_log_io,
                            pgx_lower_log_trace, newval);
}

static void register_bool_guc(const char *name, const char *desc, bool *var) {
    DefineCustomBoolVariable(name, desc, NULL, var, false, PGC_USERSET, 0, NULL, update_logging_bool, NULL);
}

static void register_string_guc(const char *name, const char *desc, char **var) {
    DefineCustomStringVariable(name, desc, NULL, var, "", PGC_USERSET, 0, NULL, update_logging_string, NULL);
}

void _PG_init(void) {
    extern void initialize_stderr_redirect(void);
    initialize_stderr_redirect();
    PGX_NOTICE_C("Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;

    g_extension_after_load = true;
    PGX_NOTICE_C("LOAD detection flag set - memory context protection enabled");

    register_bool_guc("pgx_lower.log_enable", "Enable PGX-Lower logging", &pgx_lower_log_enable);
    register_bool_guc("pgx_lower.log_debug", "Enable debug logging", &pgx_lower_log_debug);
    register_bool_guc("pgx_lower.log_ir", "Enable IR (intermediate representation) logging", &pgx_lower_log_ir);
    register_bool_guc("pgx_lower.log_io", "Enable I/O boundary logging", &pgx_lower_log_io);
    register_bool_guc("pgx_lower.log_trace", "Enable trace logging", &pgx_lower_log_trace);
    register_string_guc("pgx_lower.enabled_categories", "Comma-separated list of enabled log categories", 
                       &pgx_lower_enabled_categories);

    PGX_NOTICE_C("Initializing MLIR pass registration...");
    extern void initialize_mlir_passes(void);
    initialize_mlir_passes();
}

void _PG_fini(void) {
    PGX_NOTICE_C("Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}

extern bool execute_mlir_text(const char* mlir_text, void* dest_receiver);
extern int get_computed_results_num_columns(void);
extern int get_computed_results_num_rows(void);
extern void get_computed_result(int row, int col, void** value_out, bool* is_null_out);

PG_FUNCTION_INFO_V1(pgx_lower_test_relalg);

Datum
pgx_lower_test_relalg(PG_FUNCTION_ARGS)
{
    ReturnSetInfo* rsinfo = (ReturnSetInfo*) fcinfo->resultinfo;

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));

    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not allowed in this context")));

    const text * mlir_input = PG_GETARG_TEXT_PP(0);
    const char * mlir_text = text_to_cstring(mlir_input);

    PGX_NOTICE_C("Executing MLIR RelAlg directly...");

    bool success = execute_mlir_text(mlir_text, NULL);

    if (!success) {
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("MLIR execution failed")));
    }

    const MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    const MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);

    const TupleDesc tupdesc = CreateTupleDescCopy(rsinfo->expectedDesc);
    Tuplestorestate *tupstore = tuplestore_begin_heap(true, false, 4096);

    const int numColumns = get_computed_results_num_columns();
    const int numRows = get_computed_results_num_rows();

    for (int row = 0; row < numRows; row++) {
        Datum *values = (Datum *) palloc(numColumns * sizeof(Datum));
        bool *nulls = (bool *) palloc(numColumns * sizeof(bool));

        for (int col = 0; col < numColumns; col++) {
            void* datum_ptr = NULL;
            bool is_null = false;
            get_computed_result(row, col, &datum_ptr, &is_null);
            values[col] = (Datum)(uintptr_t)datum_ptr;
            nulls[col] = is_null;
        }

        tuplestore_putvalues(tupstore, tupdesc, values, nulls);

        pfree(values);
        pfree(nulls);
    }

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;

    MemoryContextSwitchTo(oldcontext);

    PG_RETURN_NULL();
}
