LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS table_a;
DROP TABLE IF EXISTS table_b;
DROP TABLE IF EXISTS table_c;

CREATE TABLE table_a
(
    id    INTEGER,
    value INTEGER,
    name  TEXT
);

CREATE TABLE table_b
(
    id     INTEGER,
    amount INTEGER,
    status TEXT
);

CREATE TABLE table_c
(
    id       INTEGER,
    category INTEGER,
    flag     INTEGER
);

INSERT INTO table_a(id, value, name)
VALUES (1, 100, 'apple'),
       (2, 200, 'banana'),
       (3, 300, 'cherry'),
       (4, 400, 'date');

INSERT INTO table_b(id, amount, status)
VALUES (1, 10, 'active'),
       (2, 20, 'pending'),
       (3, 30, 'active'),
       (5, 50, 'inactive');

INSERT INTO table_c(id, category, flag)
VALUES (1, 1, 1),
       (2, 1, 0),
       (3, 2, 1),
       (4, 2, 0);

SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER,RUNTIME,JIT,GENERAL';

SELECT table_a.name, table_a.value + table_b.amount AS total
FROM table_a, table_b
WHERE table_a.id = table_b.id;