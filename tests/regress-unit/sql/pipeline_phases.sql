\set ON_ERROR_STOP on

CREATE SCHEMA IF NOT EXISTS test;

CREATE OR REPLACE FUNCTION test.pipeline_mapop_print() RETURNS VOID
    AS '$libdir/pgx_lower', 'ts_test_pipeline_mapop_print' LANGUAGE C VOLATILE;

DO $$ BEGIN
    PERFORM test.pipeline_mapop_print();
END $$;

DROP SCHEMA test CASCADE;
