#!/bin/bash
set -e

if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
    echo "Initializing PostgreSQL database..."
    su - postgres -c '/usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data'
fi

echo "Configuring PostgreSQL for external connections..."
sed -i "s/^#*listen_addresses.*/listen_addresses = '*'/" /var/lib/postgresql/data/postgresql.conf

grep -q 'host all all 0.0.0.0/0 trust' /var/lib/postgresql/data/pg_hba.conf || \
    echo 'host all all 0.0.0.0/0 trust' >> /var/lib/postgresql/data/pg_hba.conf

chown postgres:postgres /var/lib/postgresql/data/postgresql.conf /var/lib/postgresql/data/pg_hba.conf

exec "$@"
