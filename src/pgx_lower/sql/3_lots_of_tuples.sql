LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS test;
CREATE TABLE test(id SERIAL);

-- Insert values 1 through 5000
INSERT INTO test(id)
SELECT generate_series(1, 5000);

SELECT * FROM test;
