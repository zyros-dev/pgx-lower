#!/bin/bash
#
# Entrypoint for pgx-lower benchmark container
# Starts PostgreSQL with benchmark-optimized configuration
#

set -e

echo -e "pgx-lower Benchmark Container"
echo "PostgreSQL Version: $(postgres --version)"
echo "Extension: pgx_lower"
echo ""

if [ "$(id -u)" = "0" ]; then
    if [ -f /var/lib/postgresql/postgresql.conf ]; then
        echo -e "Using benchmark-optimized PostgreSQL configuration"
        cp /var/lib/postgresql/postgresql.conf /var/lib/postgresql/data/postgresql.conf
    fi

    echo "host all all 0.0.0.0/0 trust" > /var/lib/postgresql/data/pg_hba.conf
    echo "host all all ::0/0 trust" >> /var/lib/postgresql/data/pg_hba.conf
    echo "local all all trust" >> /var/lib/postgresql/data/pg_hba.conf

    chown -R postgres:postgres /var/lib/postgresql/data

    export PGDATA=/var/lib/postgresql/data

    if [ "$1" = "postgres" ]; then
        exec gosu postgres postgres -D "$PGDATA"
    else
        exec gosu postgres "$@"
    fi
fi

exec "$@"
