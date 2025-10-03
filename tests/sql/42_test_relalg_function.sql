LOAD 'pgx_lower.so';

-- Create a simple test table
DROP TABLE IF EXISTS test_relalg;
CREATE TABLE test_relalg (id INTEGER, value INTEGER);
INSERT INTO test_relalg VALUES (1, 100), (2, 200), (3, 300);

-- Test direct RelAlg execution: simple table scan
SELECT * FROM pgx_lower_test_relalg(
  format($$
module {
  func.func @main() -> !dsa.table {
    %%0 = relalg.basetable {
      column_order = ["id", "value"],
      table_identifier = "test_relalg|oid:%s"
    } columns: {
      id => @test_relalg::@id({type = i32}),
      value => @test_relalg::@value({type = i32})
    }
    %%1 = relalg.materialize %%0 [@test_relalg::@id, @test_relalg::@value] => ["id", "value"] : !dsa.table
    return %%1 : !dsa.table
  }
}
$$, (SELECT oid FROM pg_class WHERE relname = 'test_relalg'))
) AS t(id INTEGER, value INTEGER);

DROP TABLE test_relalg;
