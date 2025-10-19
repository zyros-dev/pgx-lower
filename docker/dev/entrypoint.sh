#!/bin/bash

if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
    echo "Initializing PostgreSQL database..."
    su - postgres -c "/usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data"

    echo "listen_addresses = '*'" >> /var/lib/postgresql/data/postgresql.conf
    echo "host all all 0.0.0.0/0 trust" >> /var/lib/postgresql/data/pg_hba.conf
fi

su - postgres -c "/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start" || \
su - postgres -c "/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile restart"

echo "PostgreSQL started"
echo "Dev container ready - workspace mounted at /workspace"

exec "$@"
