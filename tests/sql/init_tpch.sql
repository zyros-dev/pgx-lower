DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS region CASCADE;
DROP TABLE IF EXISTS nation CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS lineitem CASCADE;

CREATE TABLE region (
                        r_regionkey INTEGER      NOT NULL,
                        r_name      CHAR(25)     NOT NULL,
                        r_comment   VARCHAR(152) NOT NULL,
                        PRIMARY KEY (r_regionkey)
);

CREATE TABLE nation (
                        n_nationkey INTEGER      NOT NULL,
                        n_name      CHAR(25)     NOT NULL,
                        n_regionkey INTEGER      NOT NULL,
                        n_comment   VARCHAR(152) NOT NULL,
                        PRIMARY KEY (n_nationkey)
);

CREATE TABLE part (
                      p_partkey     INTEGER        NOT NULL,
                      p_name        VARCHAR(55)    NOT NULL,
                      p_mfgr        CHAR(25)       NOT NULL,
                      p_brand       CHAR(10)       NOT NULL,
                      p_type        VARCHAR(25)    NOT NULL,
                      p_size        INTEGER        NOT NULL,
                      p_container   CHAR(10)       NOT NULL,
                      p_retailprice DECIMAL(12, 2) NOT NULL,
                      p_comment     VARCHAR(23)    NOT NULL,
                      PRIMARY KEY (p_partkey)
);

CREATE TABLE supplier (
                          s_suppkey   INTEGER        NOT NULL,
                          s_name      CHAR(25)       NOT NULL,
                          s_address   VARCHAR(40)    NOT NULL,
                          s_nationkey INTEGER        NOT NULL,
                          s_phone     CHAR(15)       NOT NULL,
                          s_acctbal   DECIMAL(12, 2) NOT NULL,
                          s_comment   VARCHAR(101)   NOT NULL,
                          PRIMARY KEY (s_suppkey)
);

CREATE TABLE partsupp (
                          ps_partkey    INTEGER        NOT NULL,
                          ps_suppkey    INTEGER        NOT NULL,
                          ps_availqty   INTEGER        NOT NULL,
                          ps_supplycost DECIMAL(12, 2) NOT NULL,
                          ps_comment    VARCHAR(199)   NOT NULL,
                          PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
                          c_custkey    INTEGER        NOT NULL,
                          c_name       VARCHAR(25)    NOT NULL,
                          c_address    VARCHAR(40)    NOT NULL,
                          c_nationkey  INTEGER        NOT NULL,
                          c_phone      CHAR(15)       NOT NULL,
                          c_acctbal    DECIMAL(12, 2) NOT NULL,
                          c_mktsegment CHAR(10)       NOT NULL,
                          c_comment    VARCHAR(117)   NOT NULL,
                          PRIMARY KEY (c_custkey)
);

CREATE TABLE orders (
                        o_orderkey      INTEGER        NOT NULL,
                        o_custkey       INTEGER        NOT NULL,
                        o_orderstatus   CHAR(1)        NOT NULL,
                        o_totalprice    DECIMAL(12, 2) NOT NULL,
                        o_orderdate     DATE           NOT NULL,
                        o_orderpriority CHAR(15)       NOT NULL,
                        o_clerk         CHAR(15)       NOT NULL,
                        o_shippriority  INTEGER        NOT NULL,
                        o_comment       VARCHAR(79)    NOT NULL,
                        PRIMARY KEY (o_orderkey)
);

CREATE TABLE lineitem (
                          l_orderkey      INTEGER        NOT NULL,
                          l_partkey       INTEGER        NOT NULL,
                          l_suppkey       INTEGER        NOT NULL,
                          l_linenumber    INTEGER        NOT NULL,
                          l_quantity      DECIMAL(12, 2) NOT NULL,
                          l_extendedprice DECIMAL(12, 2) NOT NULL,
                          l_discount      DECIMAL(12, 2) NOT NULL,
                          l_tax           DECIMAL(12, 2) NOT NULL,
                          l_returnflag    CHAR(1)        NOT NULL,
                          l_linestatus    CHAR(1)        NOT NULL,
                          l_shipdate      DATE           NOT NULL,
                          l_commitdate    DATE           NOT NULL,
                          l_receiptdate   DATE           NOT NULL,
                          l_shipinstruct  CHAR(25)       NOT NULL,
                          l_shipmode      CHAR(10)       NOT NULL,
                          l_comment       VARCHAR(44)    NOT NULL,
                          PRIMARY KEY (l_orderkey, l_linenumber)
);

INSERT INTO region(r_regionkey, r_name, r_comment)
VALUES (0, 'AFRICA', 'Region comment for Africa'),
       (1, 'AMERICA', 'Region comment for America'),
       (2, 'ASIA', 'Region comment for Asia'),
       (3, 'EUROPE', 'Region comment for Europe'),
       (4, 'MIDDLE EAST', 'Region comment for Middle East');

INSERT INTO nation(n_nationkey, n_name, n_regionkey, n_comment)
VALUES (0, 'ALGERIA', 0, 'Nation Algeria'),
       (1, 'BRAZIL', 1, 'Nation Brazil'),
       (2, 'CANADA', 1, 'Nation Canada'),
       (3, 'FRANCE', 3, 'Nation France'),
       (4, 'GERMANY', 3, 'Nation Germany'),
       (5, 'INDIA', 2, 'Nation India'),
       (6, 'JAPAN', 2, 'Nation Japan'),
       (7, 'SAUDI ARABIA', 4, 'Nation Saudi Arabia');

INSERT INTO supplier(s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment)
VALUES (1, 'Supplier#000000001', 'Address 1', 0, '27-918-335-1736', 5755.94, 'Comment 1'),
       (2, 'Supplier#000000002', 'Address 2', 3, '16-768-687-3665', 4032.68, 'Comment 2'),
       (3, 'Supplier#000000003', 'Address 3', 4, '17-369-536-1112', 2972.26, 'Comment 3 Customer Complaints'),
       (4, 'Supplier#000000004', 'Address 4', 2, '20-469-856-8873', 4641.08, 'Comment 4'),
       (5, 'Supplier#000000005', 'Address 5', 7, '30-114-968-4951', 1337.45, 'Comment 5'),
       (6, 'Supplier#000000006', 'Address 6', 2, '31-222-333-4444', 3500.00, 'Query 20 CANADA supplier'),
       (7, 'Supplier#000000007', 'Address 7', 5, '91-555-0107', 2500.00, 'Query 5 INDIA supplier');

INSERT INTO part(p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment)
VALUES (1, 'forest green puff', 'Manufacturer#1', 'Brand#12', 'PROMO BURNISHED COPPER', 5, 'SM BOX', 901.00, 'Part comment 1'),
       (2, 'azure blue metal', 'Manufacturer#2', 'Brand#23', 'MEDIUM POLISHED BRASS', 15, 'MED BOX', 902.00, 'Part comment 2'),
       (3, 'spring wheat linen', 'Manufacturer#3', 'Brand#34', 'ECONOMY ANODIZED STEEL', 23, 'LG CASE', 903.00, 'Part comment 3'),
       (4, 'powder almond thistle', 'Manufacturer#4', 'Brand#45', 'STANDARD POLISHED TIN', 45, 'LG PKG', 904.00, 'Part comment 4'),
       (5, 'forest green steel', 'Manufacturer#5', 'Brand#12', 'PROMO PLATED COPPER', 3, 'SM CASE', 905.00, 'Part comment 5');

INSERT INTO partsupp(ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment)
VALUES (1, 1, 3325, 771.64, 'Partsupp comment 1'),
       (1, 2, 8076, 993.49, 'Partsupp comment 2'),
       (2, 2, 4969, 337.09, 'Partsupp comment 3'),
       (2, 3, 8539, 350.63, 'Partsupp comment 4'),
       (3, 3, 4651, 438.37, 'Partsupp comment 5'),
       (3, 4, 7054, 364.48, 'Partsupp comment 6'),
       (4, 4, 6942, 113.97, 'Partsupp comment 7'),
       (5, 5, 9280, 652.89, 'Partsupp comment 8'),
       (1, 6, 5000, 800.00, 'Query 20 partsupp'),
       (5, 7, 4500, 650.00, 'Query 5 INDIA partsupp');

INSERT INTO customer(c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment)
VALUES (1, 'Customer#000000001', 'Address 1', 4, '13-555-0101', 711.56, 'BUILDING', 'Comment 1'),
       (2, 'Customer#000000002', 'Address 2', 2, '31-555-0102', 121.65, 'AUTOMOBILE', 'Comment 2'),
       (3, 'Customer#000000003', 'Address 3', 3, '23-555-0103', 7498.12, 'MACHINERY', 'Comment 3'),
       (4, 'Customer#000000004', 'Address 4', 5, '29-555-0104', 2866.83, 'HOUSEHOLD', 'Comment 4'),
       (5, 'Customer#000000005', 'Address 5', 5, '30-555-0105', 794.47, 'FURNITURE', 'Comment 5 special requests'),
       (6, 'Customer#000000006', 'Address 6', 2, '31-555-0106', 5500.00, 'BUILDING', 'Query 22 customer no orders');

INSERT INTO orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment)
VALUES (1, 1, 'O', 173665.47, '1996-01-02', '5-LOW', 'Clerk#000000951', 0, 'Order comment 1'),
       (2, 1, 'O', 46929.18, '1996-12-01', '1-URGENT', 'Clerk#000000880', 0, 'Order comment 2'),
       (3, 2, 'F', 193846.25, '1993-10-14', '5-LOW', 'Clerk#000000955', 0, 'Order comment 3'),
       (4, 3, 'O', 32151.78, '1995-10-11', '2-HIGH', 'Clerk#000000124', 0, 'Order comment 4'),
       (5, 4, 'F', 144659.20, '1994-07-30', '5-LOW', 'Clerk#000000925', 0, 'Order comment 5'),
       (6, 5, 'F', 58749.59, '1992-02-21', '4-NOT SPECIFIED', 'Clerk#000000058', 0, 'Order comment 6 special requests'),
       (7, 1, 'O', 85000.00, '1995-03-10', '3-MEDIUM', 'Clerk#000000100', 1, 'Query 3 order'),
       (8, 1, 'F', 95000.00, '1993-08-15', '1-URGENT', 'Clerk#000000200', 0, 'Query 4 order'),
       (9, 4, 'O', 75000.00, '1994-06-20', '2-HIGH', 'Clerk#000000300', 0, 'Query 5 ASIA order'),
       (10, 2, 'O', 105000.00, '1995-06-15', '5-LOW', 'Clerk#000000400', 0, 'Query 8 order'),
       (11, 4, 'F', 65000.00, '1994-02-10', '1-URGENT', 'Clerk#000000500', 0, 'Query 21 order'),
       (12, 4, 'O', 55000.00, '1994-03-15', '2-HIGH', 'Clerk#000000600', 0, 'Query 5 INDIA order'),
       (13, 5, 'F', 45000.00, '1994-01-20', '1-URGENT', 'Clerk#000000700', 0, 'Query 21 Saudi order');

INSERT INTO lineitem(l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment)
VALUES (1, 1, 1, 1, 17.00, 21168.23, 0.04, 0.02, 'N', 'O', '1996-03-13', '1996-02-12', '1996-03-22', 'DELIVER IN PERSON', 'TRUCK', 'Lineitem comment 1'),
       (1, 2, 2, 2, 36.00, 45983.16, 0.09, 0.06, 'N', 'O', '1996-04-12', '1996-02-28', '1996-04-20', 'TAKE BACK RETURN', 'MAIL', 'Lineitem comment 2'),
       (2, 3, 3, 1, 38.00, 42653.00, 0.00, 0.05, 'R', 'F', '1997-01-28', '1997-01-14', '1997-02-02', 'TAKE BACK RETURN', 'RAIL', 'Lineitem comment 3'),
       (3, 1, 1, 1, 45.00, 54058.05, 0.06, 0.00, 'R', 'F', '1994-02-02', '1994-01-04', '1994-02-23', 'NONE', 'AIR', 'Lineitem comment 4'),
       (3, 4, 4, 2, 49.00, 46796.47, 0.10, 0.00, 'R', 'F', '1993-11-09', '1993-12-20', '1993-11-24', 'TAKE BACK RETURN', 'SHIP', 'Lineitem comment 5'),
       (4, 2, 2, 1, 30.00, 55380.00, 0.03, 0.08, 'N', 'O', '1995-11-15', '1995-10-10', '1995-11-24', 'DELIVER IN PERSON', 'AIR REG', 'Lineitem comment 6'),
       (5, 3, 3, 1, 20.00, 30780.20, 0.04, 0.03, 'R', 'F', '1994-08-15', '1994-08-08', '1994-08-25', 'NONE', 'MAIL', 'Lineitem comment 7'),
       (5, 5, 5, 2, 10.00, 18050.00, 0.07, 0.02, 'A', 'F', '1994-10-16', '1994-09-25', '1994-10-20', 'DELIVER IN PERSON', 'SHIP', 'Lineitem comment 8'),
       (2, 1, 1, 3, 50.00, 60000.00, 0.05, 0.01, 'N', 'O', '1996-02-15', '1996-01-20', '1996-03-01', 'DELIVER IN PERSON', 'AIR', 'Query 3,8 lineitem'),
       (4, 3, 3, 2, 100.00, 120000.00, 0.02, 0.04, 'N', 'O', '1995-12-01', '1995-11-01', '1995-12-15', 'NONE', 'SHIP', 'Query 12 lineitem 1'),
       (5, 1, 1, 3, 150.00, 180000.00, 0.01, 0.03, 'N', 'O', '1994-08-01', '1994-07-15', '1994-07-20', 'DELIVER IN PERSON', 'MAIL', 'Query 4,12 lineitem'),
       (6, 2, 2, 1, 200.00, 240000.00, 0.03, 0.02, 'R', 'F', '1992-03-10', '1992-03-01', '1992-03-20', 'NONE', 'TRUCK', 'Query 18 lineitem 1'),
       (6, 3, 3, 2, 150.00, 180000.00, 0.04, 0.01, 'R', 'F', '1992-03-15', '1992-03-01', '1992-03-25', 'NONE', 'TRUCK', 'Query 18 lineitem 2'),
       (7, 1, 1, 1, 25.00, 30000.00, 0.05, 0.02, 'N', 'O', '1995-03-20', '1995-03-10', '1995-03-25', 'DELIVER IN PERSON', 'TRUCK', 'Query 3 lineitem'),
       (8, 2, 2, 1, 40.00, 48000.00, 0.04, 0.03, 'R', 'F', '1993-09-01', '1993-08-20', '1993-08-25', 'NONE', 'MAIL', 'Query 4 lineitem'),
       (9, 5, 5, 1, 35.00, 42000.00, 0.06, 0.01, 'N', 'O', '1994-07-15', '1994-06-25', '1994-07-20', 'TAKE BACK RETURN', 'SHIP', 'Query 5 lineitem'),
       (10, 3, 3, 1, 45.00, 54000.00, 0.03, 0.02, 'N', 'O', '1995-07-01', '1995-06-20', '1995-07-10', 'DELIVER IN PERSON', 'AIR REG', 'Query 8 lineitem'),
       (11, 4, 4, 1, 30.00, 36000.00, 0.02, 0.01, 'R', 'F', '1994-03-10', '1994-02-15', '1994-03-05', 'NONE', 'RAIL', 'Query 21 lineitem 1'),
       (11, 5, 5, 2, 28.00, 33600.00, 0.03, 0.02, 'R', 'F', '1994-03-15', '1994-02-20', '1994-03-10', 'TAKE BACK RETURN', 'TRUCK', 'Query 21 lineitem 2'),
       (12, 5, 7, 1, 40.00, 48000.00, 0.05, 0.02, 'N', 'O', '1994-04-15', '1994-03-20', '1994-04-20', 'NONE', 'TRUCK', 'Query 5 INDIA lineitem'),
       (13, 1, 5, 1, 35.00, 42000.00, 0.04, 0.01, 'R', 'F', '1994-02-10', '1994-01-25', '1994-02-05', 'NONE', 'RAIL', 'Query 21 Saudi lineitem 1'),
       (13, 2, 7, 2, 30.00, 36000.00, 0.03, 0.02, 'R', 'F', '1994-02-15', '1994-02-20', '1994-02-10', 'TAKE BACK RETURN', 'TRUCK', 'Query 21 Saudi lineitem 2'),
       (5, 2, 2, 4, 15.00, 18000.00, 0.06, 0.01, 'N', 'O', '1994-07-25', '1994-08-05', '1994-08-10', 'NONE', 'MAIL', 'Query 12 MAIL lineitem'),
       (11, 3, 3, 3, 20.00, 24000.00, 0.05, 0.02, 'R', 'F', '1994-02-15', '1994-02-20', '1994-02-25', 'DELIVER IN PERSON', 'SHIP', 'Query 12 SHIP lineitem'),
       (12, 1, 6, 2, 22.00, 26400.00, 0.04, 0.01, 'N', 'O', '1994-04-10', '1994-03-25', '1994-04-15', 'NONE', 'TRUCK', 'Query 20 forest lineitem'),
       (7, 5, 5, 2, 8.00, 9600.00, 0.03, 0.01, 'N', 'O', '1995-09-15', '1995-09-10', '1995-09-20', 'NONE', 'TRUCK', 'Query 14 Sept 1995'),
       (10, 2, 2, 2, 5.00, 6000.00, 0.02, 0.01, 'N', 'O', '1995-07-10', '1995-07-05', '1995-07-15', 'NONE', 'RAIL', 'Query 17 Brand23 MED BOX'),
       (2, 5, 5, 4, 8.00, 9600.00, 0.04, 0.02, 'N', 'O', '1996-03-10', '1996-03-05', '1996-03-15', 'DELIVER IN PERSON', 'AIR', 'Query 19 Brand12 AIR');