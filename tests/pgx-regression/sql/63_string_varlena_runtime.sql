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

DROP TABLE string_varlena_runtime;
