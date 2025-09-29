LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS sort_test;
CREATE TABLE sort_test (
    id INTEGER,
    name TEXT
);

INSERT INTO sort_test VALUES
    (1, 'Zebra'),
    (2, 'Apple'),
    (3, 'Mango'),
    (4, 'Banana');

SELECT name FROM sort_test ORDER BY name;

SELECT t1.name
FROM sort_test t1
JOIN sort_test t2 ON t1.id = t2.id
ORDER BY t1.name;

DROP TABLE IF EXISTS regions;
DROP TABLE IF EXISTS countries;

CREATE TABLE regions (
    region_id INTEGER,
    region_name TEXT
);

CREATE TABLE countries (
    country_id INTEGER,
    country_name TEXT,
    region_id INTEGER
);

INSERT INTO regions VALUES
    (1, 'Asia'),
    (2, 'Europe'),
    (3, 'Americas');

INSERT INTO countries VALUES
    (1, 'India', 1),
    (2, 'China', 1),
    (3, 'France', 2),
    (4, 'Germany', 2),
    (5, 'Brazil', 3),
    (6, 'Canada', 3);

SELECT r.region_name, COUNT(*) as country_count
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY r.region_name;

SELECT c.country_name
FROM countries c
JOIN regions r1 ON c.region_id = r1.region_id
JOIN regions r2 ON r1.region_id = r2.region_id
WHERE r2.region_name = 'Asia'
ORDER BY c.country_name;

SELECT country_name
FROM (
    SELECT c.country_name, r.region_id
    FROM countries c
    JOIN regions r ON c.region_id = r.region_id
    WHERE r.region_name IN ('Asia', 'Europe')
) sub
ORDER BY country_name;

DROP TABLE IF EXISTS bpchar_test;
CREATE TABLE bpchar_test (
    id INTEGER,
    code CHAR(5)
);

INSERT INTO bpchar_test VALUES
    (1, 'ZZZ'),
    (2, 'AAA'),
    (3, 'MMM'),
    (4, 'BBB');

SELECT code FROM bpchar_test ORDER BY code;

SELECT c.country_name, SUM(c.country_id) as total
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY c.country_name
ORDER BY total DESC;

DROP TABLE IF EXISTS date_test;
CREATE TABLE date_test (
    id INTEGER,
    event_date DATE
);

INSERT INTO date_test VALUES
    (1, '2024-12-31'),
    (2, '2024-01-01'),
    (3, '2024-06-15'),
    (4, '2024-03-20');

SELECT event_date FROM date_test ORDER BY event_date;

DROP TABLE IF EXISTS timestamp_test;
CREATE TABLE timestamp_test (
    id INTEGER,
    event_time TIMESTAMP
);

INSERT INTO timestamp_test VALUES
    (1, '2024-12-31 23:59:59'),
    (2, '2024-01-01 00:00:00'),
    (3, '2024-06-15 12:30:00'),
    (4, '2024-03-20 08:15:30');

SELECT event_time FROM timestamp_test ORDER BY event_time;

SELECT r.region_name,
       COUNT(*) as country_count,
       SUM(c.country_id) as total_ids
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY total_ids DESC, r.region_name;

SELECT r.region_name, COUNT(*) as cnt
FROM regions r
JOIN regions r2 ON r.region_id = r2.region_id
GROUP BY r.region_name
ORDER BY r.region_name;

DROP TABLE IF EXISTS mixed_sort_test;
CREATE TABLE mixed_sort_test (
    id INTEGER,
    name TEXT,
    value DECIMAL(10, 2)
);

INSERT INTO mixed_sort_test VALUES
    (1, 'Alpha', 100.50),
    (2, 'Beta', 200.75),
    (3, 'Alpha', 150.25),
    (4, 'Gamma', 100.50),
    (5, 'Beta', 100.50);

SELECT value, name
FROM mixed_sort_test
ORDER BY value DESC, name ASC;

SELECT r.region_name,
       COUNT(*) as cnt,
       SUM(c.country_id) as total
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY cnt DESC, r.region_name ASC;

SELECT value, name
FROM mixed_sort_test
ORDER BY value DESC, name DESC;

SELECT r.region_name,
       c.country_name,
       c.country_id
FROM countries c
JOIN regions r ON c.region_id = r.region_id
ORDER BY r.region_name ASC, country_id DESC, c.country_name ASC;

SELECT EXTRACT(year FROM event_date) as year, COUNT(*) as cnt
FROM date_test
GROUP BY EXTRACT(year FROM event_date)
ORDER BY year;

SELECT
    CASE
        WHEN c.country_id < 3 THEN 'Low'
        WHEN c.country_id < 5 THEN 'Medium'
        ELSE 'High'
    END as category,
    COUNT(*) as cnt
FROM countries c
GROUP BY category
ORDER BY category;

SELECT r.region_name,
       SUM(c.country_id) as total,
       COUNT(*) as cnt,
       SUM(c.country_id) / COUNT(*) as avg_id
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY avg_id DESC;

SELECT year, total_countries
FROM (
    SELECT EXTRACT(year FROM event_date) as year,
           COUNT(*) as total_countries
    FROM date_test
    GROUP BY EXTRACT(year FROM event_date)
) sub
ORDER BY year;


SELECT r.region_name,
       SUM(CASE WHEN c.country_id < 4 THEN c.country_id ELSE 0 END) as conditional_sum,
       SUM(c.country_id) as total_sum
FROM countries c
JOIN regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY r.region_name;

SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER,RUNTIME,JIT,GENERAL';

SELECT sub.region_name,
       sub.conditional_sum / sub.total_sum as ratio
FROM (
    SELECT r.region_name,
           SUM(CASE WHEN c.country_id < 4 THEN c.country_id ELSE 0 END) as conditional_sum,
           SUM(c.country_id) as total_sum
    FROM countries c
    JOIN regions r ON c.region_id = r.region_id
    GROUP BY r.region_name
) sub
ORDER BY ratio DESC;

DROP TABLE sort_test;
DROP TABLE countries;
DROP TABLE regions;
DROP TABLE bpchar_test;
DROP TABLE date_test;
DROP TABLE timestamp_test;
DROP TABLE decimal_test;
DROP TABLE mixed_sort_test;