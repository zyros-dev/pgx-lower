#!/bin/bash

set -e

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "usage: $0 <iterations> <scale_factor> [doubling_loops]"
    echo "example: $0 3 0.01"
    echo "example: $0 3 0.01 7  # run 3 times, then double SF 7 more times"
    exit 1
fi

iters=$1
sf=$2
doubling=${3:-0}

run_test_batch() {
    local current_sf=$1
    local batch_iters=$2

    echo "generating tpch data (SF=$current_sf)..."
    make tpch-data SF=$current_sf
    echo

    for i in $(seq 1 $batch_iters); do
        echo "=== iteration $i/$batch_iters (SF=$current_sf) ==="

        start_time=$(date +%s)
        tmpfile=$(mktemp)
        make ptest > $tmpfile 2>&1
        end_time=$(date +%s)

        # check for "All X tables match" in the last few lines
        if tail -10 $tmpfile | grep -qE "All [0-9]+ tables match"; then
            match=$(tail -10 $tmpfile | grep -oE "All [0-9]+ tables match")
            echo "ok: $match"

            # Calculate elapsed time
            elapsed=$((end_time - start_time))
            minutes=$((elapsed / 60))
            seconds=$((elapsed % 60))
            echo "took $minutes minutes and $seconds seconds"

            rm $tmpfile
        else
            echo "FAILED - validation message not found"
            echo "last 20 lines:"
            tail -20 $tmpfile
            echo
            echo "full output: $tmpfile"
            exit 1
        fi
        echo
    done
}

run_test_batch $sf $iters

for d in $(seq 1 $doubling); do
    sf=$(echo "$sf * 2" | bc)
    echo "--- doubling to SF=$sf ---"
    run_test_batch $sf $iters
done

echo "all iterations passed"
