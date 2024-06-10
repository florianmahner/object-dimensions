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
    a) accuracy=true ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

if $jackknife; then
    echo "Run jackknife analysis"
    python experiments/human_dnn/jackknife.py --config "./configs/human_dnn_comparison.toml" --table "jackknife"
fi

if $comparison; then
    echo "Run comparison analysis"
    python experiments/human_dnn/compare_modalities.py --config "./configs/human_dnn_comparison.toml"  --table "comparison"
fi

if $accuracy; then
  echo "Run Reconstruction Accuracy RSMs"
  python experiments/human_dnn/reconstruction_accuracy.py --config "./configs/human_dnn_comparison.toml"
fi

if $rsa; then
  echo "Run RSA analysis"
  python experiments/rsa/run_rsa_analysis.py --config "./configs/human_dnn_comparison.toml" --table "rsa"
fi