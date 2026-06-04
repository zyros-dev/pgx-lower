LOAD 'pgx_lower.so';

CREATE TABLE route_notice_smoke(id int4);
INSERT INTO route_notice_smoke VALUES (1), (2);

SET pgx_lower.execution_mode = 'force_fallback';
SELECT id FROM route_notice_smoke ORDER BY id;

SET pgx_lower.execution_mode = 'auto';
SELECT id FROM route_notice_smoke ORDER BY id;

SET pgx_lower.execution_mode = 'auto';
SELECT generate_series(1, 2);

SET pgx_lower.execution_mode = 'bogus';

DROP TABLE route_notice_smoke;
