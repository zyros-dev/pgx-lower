#!/bin/bash
sudo pkill -u postgres postgres
#sudo lsof -t -i :1234 | xargs -r sudo kill -9
# To debug with gdbserver, uncomment the following line:
# sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH gdbserver --debug localhost:1234 /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data
ulimit -c unlimited
sudo -u postgres ulimit -c unlimited
# Launch postgres with the correct LD_LIBRARY_PATH for both MLIR and LLVM
# sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH LD_DEBUG=libs /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data
sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data
