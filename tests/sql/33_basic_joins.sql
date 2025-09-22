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

SELECT *
FROM table_a, table_b
WHERE table_a.id = table_b.id;

SELECT table_a.name, table_b.amount
FROM table_a, table_b
WHERE table_a.id = table_b.id;

SELECT table_a.name, table_b.amount, table_b.status
FROM table_a, table_b
WHERE table_a.id = table_b.id AND table_b.status = 'active';

SELECT table_a.name, table_b.amount, table_c.category
FROM table_a, table_b, table_c
WHERE table_a.id = table_b.id AND table_a.id = table_c.id;

SELECT table_a.name, table_a.value, table_b.amount
FROM table_a, table_b
WHERE table_a.id = table_b.id AND table_a.value > table_b.amount * 5;

SELECT table_a.name, table_b.amount
FROM table_a, table_b
WHERE table_a.id = table_b.id
ORDER BY table_b.amount DESC;

SELECT table_a.id AS a_id, table_b.id AS b_id
FROM table_a, table_b
ORDER BY a_id, b_id;

SELECT table_a.name, table_a.value + table_b.amount AS total
FROM table_a, table_b
WHERE table_a.id = table_b.id;

SELECT table_c.category, SUM(table_a.value) AS total_value
FROM table_a, table_c
WHERE table_a.id = table_c.id
GROUP BY table_c.category
ORDER BY table_c.category;

SELECT a1.name AS name1, a2.name AS name2
FROM table_a a1, table_a a2
WHERE a1.value < a2.value AND a1.id < a2.id
ORDER BY a1.name, a2.name;

DROP TABLE table_c;
DROP TABLE table_b;
DROP TABLE table_a;