LOAD 'pgx_lower.so';

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
       (5, 'Supplier#000000005', 'Address 5', 7, '30-114-968-4951', 1337.45, 'Comment 5');

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
       (5, 5, 9280, 652.89, 'Partsupp comment 8');

INSERT INTO customer(c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment)
VALUES (1, 'Customer#000000001', 'Address 1', 4, '13-555-0101', 711.56, 'BUILDING', 'Comment 1'),
       (2, 'Customer#000000002', 'Address 2', 2, '31-555-0102', 121.65, 'AUTOMOBILE', 'Comment 2'),
       (3, 'Customer#000000003', 'Address 3', 3, '23-555-0103', 7498.12, 'MACHINERY', 'Comment 3'),
       (4, 'Customer#000000004', 'Address 4', 4, '29-555-0104', 2866.83, 'HOUSEHOLD', 'Comment 4'),
       (5, 'Customer#000000005', 'Address 5', 5, '30-555-0105', 794.47, 'FURNITURE', 'Comment 5 special requests');

INSERT INTO orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment)
VALUES (1, 1, 'O', 173665.47, '1996-01-02', '5-LOW', 'Clerk#000000951', 0, 'Order comment 1'),
       (2, 1, 'O', 46929.18, '1996-12-01', '1-URGENT', 'Clerk#000000880', 0, 'Order comment 2'),
       (3, 2, 'F', 193846.25, '1993-10-14', '5-LOW', 'Clerk#000000955', 0, 'Order comment 3'),
       (4, 3, 'O', 32151.78, '1995-10-11', '2-HIGH', 'Clerk#000000124', 0, 'Order comment 4'),
       (5, 4, 'F', 144659.20, '1994-07-30', '5-LOW', 'Clerk#000000925', 0, 'Order comment 5'),
       (6, 5, 'F', 58749.59, '1992-02-21', '4-NOT SPECIFIED', 'Clerk#000000058', 0, 'Order comment 6 special requests');

INSERT INTO lineitem(l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment)
VALUES (1, 1, 1, 1, 17.00, 21168.23, 0.04, 0.02, 'N', 'O', '1996-03-13', '1996-02-12', '1996-03-22', 'DELIVER IN PERSON', 'TRUCK', 'Lineitem comment 1'),
       (1, 2, 2, 2, 36.00, 45983.16, 0.09, 0.06, 'N', 'O', '1996-04-12', '1996-02-28', '1996-04-20', 'TAKE BACK RETURN', 'MAIL', 'Lineitem comment 2'),
       (2, 3, 3, 1, 38.00, 42653.00, 0.00, 0.05, 'R', 'F', '1997-01-28', '1997-01-14', '1997-02-02', 'TAKE BACK RETURN', 'RAIL', 'Lineitem comment 3'),
       (3, 1, 1, 1, 45.00, 54058.05, 0.06, 0.00, 'R', 'F', '1994-02-02', '1994-01-04', '1994-02-23', 'NONE', 'AIR', 'Lineitem comment 4'),
       (3, 4, 4, 2, 49.00, 46796.47, 0.10, 0.00, 'R', 'F', '1993-11-09', '1993-12-20', '1993-11-24', 'TAKE BACK RETURN', 'SHIP', 'Lineitem comment 5'),
       (4, 2, 2, 1, 30.00, 55380.00, 0.03, 0.08, 'N', 'O', '1995-11-15', '1995-10-10', '1995-11-24', 'DELIVER IN PERSON', 'AIR REG', 'Lineitem comment 6'),
       (5, 3, 3, 1, 20.00, 30780.20, 0.04, 0.03, 'R', 'F', '1994-08-15', '1994-08-08', '1994-08-25', 'NONE', 'MAIL', 'Lineitem comment 7'),
       (5, 5, 5, 2, 10.00, 18050.00, 0.07, 0.02, 'A', 'F', '1994-10-16', '1994-09-25', '1994-10-20', 'DELIVER IN PERSON', 'SHIP', 'Lineitem comment 8');

-- TPC-H Query 1
select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus;

-- TPC-H Query 2
select
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
from
        part,
        supplier,
        partsupp,
        nation,
        region
where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and p_size = 15
        and p_type like '%BRASS'
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'EUROPE'
        and ps_supplycost = (
                select
                        min(ps_supplycost)
                from
                        partsupp,
                        supplier,
                        nation,
                        region
                where
                        p_partkey = ps_partkey
                        and s_suppkey = ps_suppkey
                        and s_nationkey = n_nationkey
                        and n_regionkey = r_regionkey
                        and r_name = 'EUROPE'
        )
order by
        s_acctbal desc,
        n_name,
        s_name,
        p_partkey
limit 100;

-- TPC-H Query 3
select
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        o_orderdate,
        o_shippriority
from
        customer,
        orders,
        lineitem
where
        c_mktsegment = 'BUILDING'
        and c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15'
        and l_shipdate > date '1995-03-15'
group by
        l_orderkey,
        o_orderdate,
        o_shippriority
order by
        revenue desc,
        o_orderdate
limit 10;

select
        o_orderpriority,
        count(*) as order_count
from
        orders
where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-10-01'
        and exists (
                select
                        *
                from
                        lineitem
                where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
        )
group by
        o_orderpriority
order by
        o_orderpriority;

-- TPC-H Query 5
select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
from
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and l_suppkey = s_suppkey
        and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'ASIA'
        and o_orderdate >= date '1994-01-01'
        and o_orderdate < date '1995-01-01'
group by
        n_name
order by
        revenue desc;

-- TPC-H Query 6
select
        sum(l_extendedprice * l_discount) as revenue
from
        lineitem
where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1995-01-01'
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l_quantity < 24;

-- TPC-H Query 7
select
        supp_nation,
        cust_nation,
        l_year,
        sum(volume) as revenue
from
        (
                select
                        n1.n_name as supp_nation,
                        n2.n_name as cust_nation,
                        extract(year from l_shipdate) as l_year,
                        l_extendedprice * (1 - l_discount) as volume
                from
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2
                where
                        s_suppkey = l_suppkey
                        and o_orderkey = l_orderkey
                        and c_custkey = o_custkey
                        and s_nationkey = n1.n_nationkey
                        and c_nationkey = n2.n_nationkey
                        and (
                                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
                        )
                        and l_shipdate between date '1995-01-01' and date '1996-12-31'
        ) as shipping
group by
        supp_nation,
        cust_nation,
        l_year
order by
        supp_nation,
        cust_nation,
        l_year;

-- TPC-H Query 8
select
        o_year,
        sum(case
                when nation = 'BRAZIL' then volume
                else 0
        end) / sum(volume) as mkt_share
from
        (
                select
                        extract(year from o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) as volume,
                        n2.n_name as nation
                from
                        part,
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2,
                        region
                where
                        p_partkey = l_partkey
                        and s_suppkey = l_suppkey
                        and l_orderkey = o_orderkey
                        and o_custkey = c_custkey
                        and c_nationkey = n1.n_nationkey
                        and n1.n_regionkey = r_regionkey
                        and r_name = 'AMERICA'
                        and s_nationkey = n2.n_nationkey
                        and o_orderdate between date '1995-01-01' and date '1996-12-31'
                        and p_type = 'ECONOMY ANODIZED STEEL'
        ) as all_nations
group by
        o_year
order by
        o_year;

-- TPC-H Query 9
select
        nation,
        o_year,
        sum(amount) as sum_profit
from
        (
                select
                        n_name as nation,
                        extract(year from o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                from
                        part,
                        supplier,
                        lineitem,
                        partsupp,
                        orders,
                        nation
                where
                        s_suppkey = l_suppkey
                        and ps_suppkey = l_suppkey
                        and ps_partkey = l_partkey
                        and p_partkey = l_partkey
                        and o_orderkey = l_orderkey
                        and s_nationkey = n_nationkey
                        and p_name like '%green%'
        ) as profit
group by
        nation,
        o_year
order by
        nation,
        o_year desc;

-- TPC-H Query 10
select
        c_custkey,
        c_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        c_acctbal,
        n_name,
        c_address,
        c_phone,
        c_comment
from
        customer,
        orders,
        lineitem,
        nation
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1994-01-01'
        and l_returnflag = 'R'
        and c_nationkey = n_nationkey
group by
        c_custkey,
        c_name,
        c_acctbal,
        c_phone,
        n_name,
        c_address,
        c_comment
order by
        revenue desc
limit 20;

-- TPC-H Query 11
select
        ps_partkey,
        sum(ps_supplycost * ps_availqty) as "value"
from
        partsupp,
        supplier,
        nation
where
        ps_suppkey = s_suppkey
        and s_nationkey = n_nationkey
        and n_name = 'GERMANY'
group by
        ps_partkey having
                sum(ps_supplycost * ps_availqty) > (
                        select
                                sum(ps_supplycost * ps_availqty) * 0.0001
                        from
                                partsupp,
                                supplier,
                                nation
                        where
                                ps_suppkey = s_suppkey
                                and s_nationkey = n_nationkey
                                and n_name = 'GERMANY'
                )
order by
        "value" desc;

-- TPC-H Query 12
select
        l_shipmode,
        sum(case
                when o_orderpriority = '1-URGENT'
                        or o_orderpriority = '2-HIGH'
                        then 1
                else 0
        end) as high_line_count,
        sum(case
                when o_orderpriority <> '1-URGENT'
                        and o_orderpriority <> '2-HIGH'
                        then 1
                else 0
        end) as low_line_count
from
        orders,
        lineitem
where
        o_orderkey = l_orderkey
        and l_shipmode in ('MAIL', 'SHIP')
        and l_commitdate < l_receiptdate
        and l_shipdate < l_commitdate
        and l_receiptdate >= date '1994-01-01'
        and l_receiptdate < date '1995-01-01'
group by
        l_shipmode
order by
        l_shipmode;

-- TPC-H Query 13
select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey) c_count
                from
                        customer left outer join orders on
                                c_custkey = o_custkey
                                and o_comment not like '%special%requests%'
                group by
                        c_custkey
        ) as c_orders
group by
        c_count
order by
        custdist desc,
        c_count desc;

-- TPC-H Query 14
select
        100.00 * sum(case
                when p_type like 'PROMO%'
                        then l_extendedprice * (1 - l_discount)
                else 0
        end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
        lineitem,
        part
where
        l_partkey = p_partkey
        and l_shipdate >= date '1995-09-01'
        and l_shipdate < date '1995-10-01';

-- TPC-H Query 15
with revenue as (
	select
		l_suppkey as supplier_no,
		sum(l_extendedprice * (1 - l_discount)) as total_revenue
	from
		lineitem
	where
		l_shipdate >= date '1996-01-01'
		and l_shipdate < date '1996-04-01'
	group by
		l_suppkey)
select
	s_suppkey,
	s_name,
	s_address,
	s_phone,
	total_revenue
from
	supplier,
	revenue
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue
	)
order by
	s_suppkey;

-- TPC-H Query 16
select
        p_brand,
        p_type,
        p_size,
        count(distinct ps_suppkey) as supplier_cnt
from
        partsupp,
        part
where
        p_partkey = ps_partkey
        and p_brand <> 'Brand#45'
        and p_type not like 'MEDIUM POLISHED%'
        and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
        and ps_suppkey not in (
                select
                        s_suppkey
                from
                        supplier
                where
                        s_comment like '%Customer%Complaints%'
        )
group by
        p_brand,
        p_type,
        p_size
order by
        supplier_cnt desc,
        p_brand,
        p_type,
        p_size;

-- TPC-H Query 17
select
        sum(l_extendedprice) / 7.0 as avg_yearly
from
        lineitem,
        part
where
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container = 'MED BOX'
        and l_quantity < (
                select
                        0.2 * avg(l_quantity)
                from
                        lineitem
                where
                        l_partkey = p_partkey
        );

-- TPC-H Query 18
select
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey having
                                sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
order by
        o_totalprice desc,
        o_orderdate
limit 100;

-- TPC-H Query 19
select
        sum(l_extendedprice* (1 - l_discount)) as revenue
from
        lineitem,
        part
where
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#12'
                and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                and l_quantity >= 1 and l_quantity <= 1 + 10
                and p_size between 1 and 5
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                and l_quantity >= 10 and l_quantity <= 10 + 10
                and p_size between 1 and 10
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#34'
                and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                and l_quantity >= 20 and l_quantity <= 20 + 10
                and p_size between 1 and 15
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        );

-- TPC-H Query 20
select
        s_name,
        s_address
from
        supplier,
        nation
where
        s_suppkey in (
                select
                        ps_suppkey
                from
                        partsupp
                where
                        ps_partkey in (
                                select
                                        p_partkey
                                from
                                        part
                                where
                                        p_name like 'forest%'
                        )
                        and ps_availqty > (
                                select
                                        0.5 * sum(l_quantity)
                                from
                                        lineitem
                                where
                                        l_partkey = ps_partkey
                                        and l_suppkey = ps_suppkey
                                        and l_shipdate >= date '1994-01-01'
                                        and l_shipdate < date '1995-01-01'
                        )
        )
        and s_nationkey = n_nationkey
        and n_name = 'CANADA'
order by
        s_name;

-- TPC-H Query 21
select
        s_name,
        count(*) as numwait
from
        supplier,
        lineitem l1,
        orders,
        nation
where
        s_suppkey = l1.l_suppkey
        and o_orderkey = l1.l_orderkey
        and o_orderstatus = 'F'
        and l1.l_receiptdate > l1.l_commitdate
        and exists (
                select
                        *
                from
                        lineitem l2
                where
                        l2.l_orderkey = l1.l_orderkey
                        and l2.l_suppkey <> l1.l_suppkey
        )
        and not exists (
                select
                        *
                from
                        lineitem l3
                where
                        l3.l_orderkey = l1.l_orderkey
                        and l3.l_suppkey <> l1.l_suppkey
                        and l3.l_receiptdate > l3.l_commitdate
        )
        and s_nationkey = n_nationkey
        and n_name = 'SAUDI ARABIA'
group by
        s_name
order by
        numwait desc,
        s_name
limit 100;

-- TPC-H Query 22
select
        cntrycode,
        count(*) as numcust,
        sum(c_acctbal) as totacctbal
from
        (
                select
                        substring(c_phone from 1 for 2) as cntrycode,
                        c_acctbal
                from
                        customer
                where
                        substring(c_phone from 1 for 2) in
                                ('13', '31', '23', '29', '30', '18', '17')
                        and c_acctbal > (
                                select
                                        avg(c_acctbal)
                                from
                                        customer
                                where
                                        c_acctbal > 0.00
                                        and substring(c_phone from 1 for 2) in
                                                ('13', '31', '23', '29', '30', '18', '17')
                        )
                        and not exists (
                                select
                                        *
                                from
                                        orders
                                where
                                        o_custkey = c_custkey
                        )
        ) as custsale
group by
        cntrycode
order by
        cntrycode;

DROP TABLE IF EXISTS lineitem CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS nation CASCADE;
DROP TABLE IF EXISTS region CASCADE;