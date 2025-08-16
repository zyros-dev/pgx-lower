LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;
DROP TABLE IF EXISTS all_types_test;

CREATE TABLE all_types_test
(
    id              SERIAL PRIMARY KEY,
    small_int       SMALLINT,
    integer_col     INTEGER,
    big_int         BIGINT,
    decimal_col     DECIMAL(10, 2),
    numeric_col     NUMERIC(10, 2),
    real_col        REAL,
    double_col      DOUBLE PRECISION,
    money_col       MONEY,
    char_col        CHAR(10),
    varchar_col     VARCHAR(255),
    text_col        TEXT,
    bytea_col       BYTEA,
    boolean_col     BOOLEAN,
    date_col        DATE,
    time_col        TIME,
    timetz_col      TIME WITH TIME ZONE,
    timestamp_col   TIMESTAMP,
    timestamptz_col TIMESTAMP WITH TIME ZONE,
    interval_col    INTERVAL,
    uuid_col        UUID,
    inet_col        INET,
    cidr_col        CIDR,
    macaddr_col     MACADDR,
    bit_col         BIT(8),
    varbit_col      VARBIT(8)
);

INSERT INTO all_types_test (small_int, integer_col, big_int, decimal_col, numeric_col,
                  real_col, double_col, money_col, char_col, varchar_col, text_col,
                  bytea_col, boolean_col, date_col, time_col, timetz_col,
                  timestamp_col, timestamptz_col, interval_col, uuid_col,
                  inet_col, cidr_col, macaddr_col, bit_col, varbit_col)
SELECT i::smallint, (1000 + i),
       (1000000000000 + i),
       (100.00 + i)::decimal(10,2), (200.00 + i)::numeric(10,2), (1.1 + i)::real, (2.2 + i)::double precision,
    (100.00 + i)::money,
    LPAD('ch' || i, 10, 'x'),
    'varchar_' || i,
    'text row ' || i,
    decode('DEADBEEF', 'hex'),
    i % 2 = 0,
    CURRENT_DATE + i,
    (TIME '12:00:00' + (i || ' minutes')::interval)::time,
    (TIME '12:00:00' + (i || ' minutes')::interval)::time with time zone,
    CURRENT_TIMESTAMP + (i || ' hours')::interval,
    CURRENT_TIMESTAMP + (i || ' hours')::interval,
    make_interval(days => i), gen_random_uuid(), ('192.168.1.' || i)::inet, '192.168.0.0/16'::cidr,
    MACADDR '08:00:2b:01:02:03', B'10101010', B'11110000'
FROM generate_series(1, 3) AS s(i);

SELECT id, small_int, integer_col, big_int FROM all_types_test;

SELECT char_col, varchar_col, text_col FROM all_types_test;

-- Skip date/time columns as they use CURRENT_DATE/CURRENT_TIMESTAMP which are non-deterministic

SELECT boolean_col, decimal_col, varchar_col FROM all_types_test;

SELECT real_col FROM all_types_test;

SELECT inet_col, bytea_col, bit_col FROM all_types_test;

-- Skip UUID column as gen_random_uuid() is non-deterministic