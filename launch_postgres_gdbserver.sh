#!/bin/bash
sudo -u postgres gdbserver --debug localhost:1234 /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data