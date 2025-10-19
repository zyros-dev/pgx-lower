#!/bin/bash
set -e

if [ "$(id -u)" = '0' ]; then
    if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
        echo "Initializing PostgreSQL database..."
        gosu postgres /usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data
    fi

    gosu postgres bash -c "sed -i \"s/^#*listen_addresses.*/listen_addresses = '*'/\" /var/lib/postgresql/data/postgresql.conf"
    gosu postgres bash -c "grep -q 'host all all 0.0.0.0/0 trust' /var/lib/postgresql/data/pg_hba.conf || echo 'host all all 0.0.0.0/0 trust' >> /var/lib/postgresql/data/pg_hba.conf"

    echo "Starting PostgreSQL..."
    exec gosu postgres /usr/local/pgsql/bin/postgres -D /var/lib/postgresql/data
else
    if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
        echo "Initializing PostgreSQL database..."
        /usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data
    fi

    sed -i "s/^#*listen_addresses.*/listen_addresses = '*'/" /var/lib/postgresql/data/postgresql.conf
    grep -q 'host all all 0.0.0.0/0 trust' /var/lib/postgresql/data/pg_hba.conf || \
        echo 'host all all 0.0.0.0/0 trust' >> /var/lib/postgresql/data/pg_hba.conf

    echo "Starting PostgreSQL..."
    exec /usr/local/pgsql/bin/postgres -D /var/lib/postgresql/data
fi
