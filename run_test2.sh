#!/usr/bin/env bash
set -euo pipefail

sum_f1=0
count=0

echo "==========================="
echo "Start Test 2: F1 Score Over 80%"
echo "==========================="

for cls in {1..19}; do
    base="./scripts/prince_seg/class${cls}_seg"

    out_test=$(bash ./scripts/prince_seg/class${cls}_seg/noise_test.sh)

    # acc from line: TEST ACC: [87.781 %]
    acc=$(printf '%s\n' "$out_test" \
    | sed -n 's/^TEST ACC: \[\([0-9.]*\) %\]/\1/p')

    # f1 from line: TEST F1: [85.929 %]
    f1=$(printf '%s\n' "$out_test" \
    | sed -n 's/^TEST F1: \[\([0-9.]*\) %\]/\1/p')

    # echo "Class $((count+1)) Accuracy = $acc"
    echo "Class $((count+1)) F1 Score = $f1"

    sum_f1=$(echo "$sum_f1 + $f1" | bc -l)
    count=$((count + 1))
done

mean_f1=$(echo "$sum_f1 / $count" | bc -l)
pass=false
if (( $(echo "$mean_f1 >= 80.0" | bc -l) )); then
    pass=true
fi

echo "==========================="
echo "Overall Mean F1 Score : $(printf "%.3f" "$mean_f1")"
if [ "$pass" = true ]; then
    echo "PASS: Mean F1 >= 80%"
else
    echo "FAIL: Mean F1 < 80%"
fi
echo "End of Test 2"
echo "==========================="