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

extern "C" {

int pg_fprintf(std::FILE* /*stream*/, const char* /*fmt*/, ...) {
    pg_stub_abort("pg_fprintf");
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
