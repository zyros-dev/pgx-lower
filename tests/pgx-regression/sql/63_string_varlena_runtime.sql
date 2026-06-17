LOAD 'pgx_lower.so';

CREATE TEMP TABLE string_varlena_runtime(
    id int4,
    t text,
    v varchar(8000),
    c char(8)
);

ALTER TABLE string_varlena_runtime ALTER COLUMN t SET STORAGE EXTERNAL;
ALTER TABLE string_varlena_runtime ALTER COLUMN v SET STORAGE EXTERNAL;

INSERT INTO string_varlena_runtime VALUES
    (1, '', '', 'A'),
    (2, 'short', 'short', 'B'),
    (3, repeat(md5('toast-a'), 300), repeat(md5('toast-b'), 200), 'C');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_63_string_varlena_runtime_pset */
\pset format csv

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_63_string_varlena_runtime_001 */
SELECT id, t, v, c, id AS witness FROM string_varlena_runtime ORDER BY id;

CREATE TEMP TABLE qa_bpchar3(id int4, c char(3));
CREATE TEMP TABLE qa_bpchar5(id int4, c char(5));
INSERT INTO qa_bpchar3 VALUES (1, 'ab'), (2, 'xy');
INSERT INTO qa_bpchar5 VALUES (10, 'ab'), (20, 'xy');

SET enable_nestloop = off;
SET enable_mergejoin = off;
SET enable_hashjoin = on;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_63_string_varlena_runtime_002 */
SELECT a.id, b.id, a.id AS witness
FROM qa_bpchar3 a
JOIN qa_bpchar5 b ON a.c = b.c
ORDER BY a.id, b.id;

RESET enable_hashjoin;
RESET enable_mergejoin;
RESET enable_nestloop;

DROP TABLE qa_bpchar3;
DROP TABLE qa_bpchar5;

DROP TABLE string_varlena_runtime;
