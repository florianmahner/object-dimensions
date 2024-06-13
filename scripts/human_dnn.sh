#!/bin/bash

# Run human_dnn experiment

jackknife=false
comparison=false
accuracy=false
rsa=false

while getopts "jcras" opt; do
  case $opt in
    c) comparison=true ;;
    j) jackknife=true ;;
    r) rsa=true ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

if $jackknife; then
    echo "Run jackknife analysis"
    python experiments/jackknife/run_jackknife.py --config "./configs/human_dnn_comparison.toml" --table "jackknife"
fi

if $comparison; then
    echo "Run comparison analysis"
    python experiments/run_comparison.py --config "./configs/human_dnn_comparison.toml"  --table "comparison"
fi


if $rsa; then
  echo "Run RSA analysis"
  python experiments/rsa/run_rsa_analysis.py --config "./configs/human_dnn_comparison.toml" --table "rsa"
fi