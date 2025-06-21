-- The purpose of this is to initialise my db connection

LOAD 'pgx_lower.so';
select pg_backend_pid();
SELECT 'hello';
CREATE TABLE test(id SERIAL);
SELECT * FROM test;

