#include "my_executor.h"
#include "executor/executor.h"

extern "C" {
#include "postgres.h"
#include "executor/tuptable.h"
#include "utils/snapmgr.h"
#include "access/htup_details.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "utils/elog.h"
#include "access/table.h"
#include "access/heapam.h"
#include "utils/rel.h"
}

bool MyCppExecutor::execute(const QueryDesc* plan) {
    if (!plan) {
        elog(ERROR, "QueryDesc is null");
        return false;
    }

    elog(NOTICE, "Inside C++ executor! Plan type: %d", plan->operation);
    elog(NOTICE, "Query text: %s", plan->sourceText ? plan->sourceText : "NULL");

    if (plan->operation != CMD_SELECT) {
        elog(NOTICE, "Not a SELECT statement, skipping");
        return false;
    }

    PlannedStmt *stmt = plan->plannedstmt;
    Plan *rootPlan = stmt->planTree;

    if (rootPlan->type != T_SeqScan) {
        elog(NOTICE, "Only simple table scans (SeqScan) are supported in raw mode.");
        return false;
    }

    SeqScan *scan = (SeqScan *)rootPlan;
    RangeTblEntry *rte = (RangeTblEntry *)list_nth(stmt->rtable, scan->scan.scanrelid - 1);
    Relation rel = table_open(rte->relid, AccessShareLock);

    TableScanDesc scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, NULL);
    TupleDesc tupdesc = RelationGetDescr(rel);
    HeapTuple tuple;

    // Print column names
    for (int i = 0; i < tupdesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        elog(NOTICE, "Column %d: %s", i + 1, NameStr(attr->attname));
    }

    // Print rows
    while ((tuple = heap_getnext(scanDesc, ForwardScanDirection)) != NULL) {
        for (int j = 0; j < tupdesc->natts; j++) {
            bool isNull;
            Datum value = heap_getattr(tuple, j + 1, tupdesc, &isNull);
            if (isNull) {
                elog(NOTICE, "Column %d: NULL", j + 1);
            } else {
                Oid typeOutput;
                bool typeIsVarlena;
                getTypeOutputInfo(TupleDescAttr(tupdesc, j)->atttypid, &typeOutput, &typeIsVarlena);
                char* str = OidOutputFunctionCall(typeOutput, value);
                elog(NOTICE, "Column %d: %s", j + 1, str);
                pfree(str);
            }
        }
        elog(NOTICE, "---");
    }

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return true;
}
