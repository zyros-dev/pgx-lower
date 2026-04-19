// PG-backend symbol stubs for the unit-test executable.
//
// src/lingodb/utility/mlir_to_postgres.cpp calls a handful of PG catalog
// and system-printf functions that only exist inside a running PG
// backend:
//
//   pg_fprintf  — libpgport's printf variant (expanded from PG macros)
//   GetDefaultOpClass / get_opclass_family / get_opfamily_member — opclass lookup
//
// Unit tests run outside PG, so those symbols are undefined at link time
// and the executable can't build. None of the unit-test paths actually
// exercise the functions that call these helpers — they test MLIR
// pipeline phases, not catalog-dependent code. So we provide stubs that
// abort loudly if any test accidentally reaches them.
//
// If a future test legitimately needs catalog lookup behaviour, the
// right move isn't to fill out these stubs — it's to refactor
// mlir_to_postgres.cpp to split the catalog-lookup bits from the
// catalog-free bits (type mapping), link only the latter into tests,
// and gate the former on a flag.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Oid = unsigned int;
using int16 = short;

[[noreturn]] static void pg_stub_abort(const char* name) {
    std::fprintf(stderr,
                 "pg_stubs: %s called in unit-test binary — this code path "
                 "isn't supposed to execute without a live PG backend. "
                 "If a test now legitimately needs PG catalog behaviour, "
                 "refactor mlir_to_postgres.cpp to split catalog-using "
                 "code from catalog-free code instead of expanding these "
                 "stubs.\n",
                 name);
    std::abort();
}

// Some of DSAToStdPatterns.cpp's hashtable-spec construction uses PG's
// MemoryContext machinery (palloc into CurTransactionContext) at MLIR
// lowering time. Unit tests don't exercise that path — they test lower-
// level phase outputs — but the symbols have to resolve for the link to
// succeed. Provide a minimal self-managed pool that either services the
// allocation (so the code can run if reached without crashing the test)
// or aborts if a path genuinely needs PG context semantics.
struct MemoryContextData;
typedef struct MemoryContextData* MemoryContext;

// Trivial storage. A test that actually allocates via these will leak
// until process exit — fine for the test's lifetime, no need for a real
// memory context implementation.

extern "C" {

// MemoryContext globals (variables, not functions).
MemoryContext CurrentMemoryContext = nullptr;
MemoryContext CurTransactionContext = nullptr;

void* MemoryContextAlloc(MemoryContext /*context*/, std::size_t size) {
    // Use malloc — test-only, no need for arena/context tracking.
    return std::malloc(size);
}

// PG's pool-allocated strdup. Same pattern as MemoryContextAlloc.
char* pstrdup(const char* s) {
    if (s == nullptr) return nullptr;
    const std::size_t len = std::strlen(s) + 1;
    char* dst = static_cast<char*>(std::malloc(len));
    std::memcpy(dst, s, len);
    return dst;
}

int pg_fprintf(std::FILE* /*stream*/, const char* /*fmt*/, ...) {
    // PG macros (PGX_WARNING, elog-family, ereport) bottom out in pg_fprintf
    // for formatted log output. In unit tests we don't care about logs —
    // silently discard. Tests that want to assert a warning fired should
    // wire their own check into the code path, not rely on log scraping.
    return 0;
}

Oid GetDefaultOpClass(Oid /*type_oid*/, Oid /*am*/) {
    pg_stub_abort("GetDefaultOpClass");
}

Oid get_opclass_family(Oid /*opclass*/) {
    pg_stub_abort("get_opclass_family");
}

Oid get_opfamily_member(Oid /*opfamily*/, Oid /*left*/, Oid /*right*/, int16 /*strategy*/) {
    pg_stub_abort("get_opfamily_member");
}

}  // extern "C"
