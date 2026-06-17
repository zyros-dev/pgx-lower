extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

extern "C" bool pgx_lower_runtime_type_oid_supported_for_testing(Oid typeOid);
extern "C" const char* pgx_lower_runtime_column_type_name_for_testing(Oid typeOid);
extern "C" const char* pgx_lower_runtime_unsupported_message_for_testing(Oid typeOid);

PGX_TEST_FN(temporal_runtime_rejects_timestamptz_oid) {
    REQUIRE(!pgx_lower_runtime_type_oid_supported_for_testing(TIMESTAMPTZOID));
    REQUIRE(pgx_lower_runtime_column_type_name_for_testing(TIMESTAMPTZOID) == nullptr);
    const char* message = pgx_lower_runtime_unsupported_message_for_testing(TIMESTAMPTZOID);
    REQUIRE(message != nullptr);
    REQUIRE(std::string(message)
            == "unsupported temporal type TIMESTAMPTZOID; timezone semantics are not supported by pgx-lower");
    PG_RETURN_VOID();
}
