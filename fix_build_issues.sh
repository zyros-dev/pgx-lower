#!/bin/bash

echo "=== Rolling back src and include directories ==="
git checkout HEAD -- ./src ./include

echo "✅ Git tree reset complete"
sleep 1

echo "🔧 Ensuring write permissions..."
chmod -R u+w ./include ./src

# ----

# Reset for a new era after manual work.


# ----

echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
echo "Claude launch a reviewer to evaluate this file really quickly after you run it!"
echo "Then after that, claude, go run \`make utest | head -2000\` and make sure none of the problems above this are in there,
and summarise the problems you do see. Specify what types of files you see problems, and how many cycles you've seen this problem in"
echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
