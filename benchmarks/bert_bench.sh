#!/bin/sh

set -x

sh run_squad.sh 0 120
sh run_squad.sh 1 1
sh run_rocblas_bench.sh  /data/squad/bert-large-uncased-seq_len=512-bs=4-steps=1/rocblas_bench.csv 580
