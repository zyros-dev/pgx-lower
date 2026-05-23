CREATE SCHEMA IF NOT EXISTS test;

CREATE OR REPLACE FUNCTION test.numeric_to_i128_zero() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_zero' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.numeric_to_i128_positive() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_positive' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.numeric_to_i128_negative() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_negative' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.numeric_to_i128_rescale_up() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_rescale_up' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.numeric_to_i128_scale_down() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_scale_down' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.numeric_to_i128_large() RETURNS VOID
    AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_large' LANGUAGE C VOLATILE;

SELECT test.numeric_to_i128_zero();
SELECT test.numeric_to_i128_positive();
SELECT test.numeric_to_i128_negative();
SELECT test.numeric_to_i128_rescale_up();
SELECT test.numeric_to_i128_scale_down();
SELECT test.numeric_to_i128_large();

DROP SCHEMA test CASCADE;
