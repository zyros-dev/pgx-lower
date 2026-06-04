extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

PGX_TEST_FN(query_analyzer_default_result_is_invalid) {
    const auto result = pgx_lower::AnalyzerResult{};
    REQUIRE(!result.isSupported());
    REQUIRE(result.reasons().size() == 1);
    REQUIRE(result.reasons().front().kind == pgx_lower::UnsupportedReasonKind::invalid);
    REQUIRE(result.primaryReasonKindName() == std::string("invalid"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_supported_result_has_no_reasons) {
    const auto result = pgx_lower::AnalyzerResult::supported();
    REQUIRE(result.isSupported());
    REQUIRE(result.reasons().empty());
    REQUIRE(result.humanSummary() == std::string("supported"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_unsupported_result_reports_first_reason) {
    auto result = pgx_lower::AnalyzerResult::unsupported(
        pgx_lower::UnsupportedReasonKind::unsupported_function,
        "unsupported function generate_series()",
        "Plan.Result.targetlist[0]");
    result.addUnsupportedReason(
        pgx_lower::UnsupportedReasonKind::unsupported_type,
        "unsupported type jsonb",
        "Plan.Result.targetlist[1]");

    REQUIRE(!result.isSupported());
    REQUIRE(result.reasons().size() == 2);
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    REQUIRE(result.primaryReasonKindName() == std::string("unsupported_function"));
    REQUIRE(result.humanSummary() == std::string(
        "unsupported_function: unsupported function generate_series() at Plan.Result.targetlist[0]"));
    PG_RETURN_VOID();
}
