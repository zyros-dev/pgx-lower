#ifndef PGX_LOWER_TEST_PGX_TEST_FN_H
#define PGX_LOWER_TEST_PGX_TEST_FN_H

#define PGX_TEST_FN(name) \
    extern "C" { \
        PG_FUNCTION_INFO_V1(ts_test_##name); \
    } \
    extern "C" Datum ts_test_##name(PG_FUNCTION_ARGS)

#endif
