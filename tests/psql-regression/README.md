# PostgreSQL Regression Burndown

This directory stores pgx-lower's imported upstream PostgreSQL regression burn-down workload.
The imported `sql/`, `expected/`, and `parallel_schedule` files are committed
so agents have a stable slow-suite surface. Generated logs and results are
ignored.

Bootstrap PostgreSQL 17.6 regression sources:

```bash
mkdir -p build-artifacts/psql-regression
curl -L https://ftp.postgresql.org/pub/source/v17.6/postgresql-17.6.tar.bz2 \
  -o build-artifacts/psql-regression/postgresql-17.6.tar.bz2
tar -xjf build-artifacts/psql-regression/postgresql-17.6.tar.bz2 \
  -C build-artifacts/psql-regression
```

Record the current known-failing baseline after reviewing the first run:

```bash
pgx-cli test psql-regression-burndown --record
```

Check delta against the committed baseline:

```bash
pgx-cli test psql-regression-burndown
```
