#!/bin/bash
sudo pkill -u postgres postgres
sudo lsof -t -i :1234 | xargs -r sudo kill -9
sudo -u postgres gdbserver --debug localhost:1234 /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data