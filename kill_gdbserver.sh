#!/bin/bash
# Find and kill all running gdbserver processes
ps aux | grep '[g]dbserver' | awk '{print $2}' | xargs -r kill -9 