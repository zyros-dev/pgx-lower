LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS decode_at_scan_t;

CREATE TABLE decode_at_scan_t
(
    id          SERIAL PRIMARY KEY,
    quantity    DECIMAL(10, 2) NOT NULL,
    price       DECIMAL(12, 2) NOT NULL,
    ship_date   DATE           NOT NULL,
    status      VARCHAR(8)     NOT NULL,
    note        TEXT           NOT NULL
);

INSERT INTO decode_at_scan_t (quantity, price, ship_date, status, note)
SELECT (i % 50 + 1)::DECIMAL(10,2),
       (1000 + i * 7)::DECIMAL(12,2),
       DATE '1997-01-01' + (i % 365),
       CASE WHEN i % 3 = 0 THEN 'A' WHEN i % 3 = 1 THEN 'B' ELSE 'C' END,
       'item-note-' || LPAD(i::TEXT, 4, '0')
FROM generate_series(1, 200) AS s(i);

-- Wide scan: read every column type on every tuple.
SELECT id, quantity, price, ship_date, status, note
FROM decode_at_scan_t
WHERE id <= 5
ORDER BY id;

-- Date-predicate filter + decimal projection across the full table.
SELECT id, quantity * price AS total, ship_date
FROM decode_at_scan_t
WHERE ship_date >= DATE '1997-12-15'
  AND ship_date <= DATE '1997-12-20'
ORDER BY id;

-- Group-by exercising decimal aggregation + string grouping.
SELECT status, sum(quantity * price) AS revenue
FROM decode_at_scan_t
WHERE ship_date >= DATE '1997-06-01'
  AND ship_date <  DATE '1997-09-01'
GROUP BY status
ORDER BY status;

-- Scan that touches a high-index column (forces walk past most attributes).
SELECT note
FROM decode_at_scan_t
WHERE id BETWEEN 100 AND 105
ORDER BY id;

DROP TABLE decode_at_scan_t;
