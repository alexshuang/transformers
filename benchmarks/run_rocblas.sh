#!/bin/sh

TOP_DIR=../
EXAMPLE=$TOP_DIR/examples/run_squad.py
BASENAME=${EXAMPLE##*/}
FNAME=${BASENAME%.*}
SQUAD_DIR=/dockerx/data/squad
MODEL_NAME=bert-large-uncased
TRAIN_FILE=train-v1.1.json
VALID_FILE=dev-v1.1.json

SEQ_LEN=512
BS=4
STEPS=3

TOOL=~/rocblas/build/release/clients/staging/rocblas-bench
OUT_DIR=$SQUAD_DIR/${MODEL_NAME}-seq_len=${SEQ_LEN}-bs=${BS}-steps=${STEPS}
ln -s ${TOOL} .

mkdir -p $OUT_DIR

export ROCBLAS_LAYER=2
export ELAPSED=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

#set -e

#rm -f ${OUT_DIR}/rocblas_kernels.csv

sed 's/\(rocblas-bench\)/\1 -i 10 -j 2/g' ${OUT_DIR}/rocblas_bench.csv > /tmp/rocblas_bench.csv
#sed 's/\(rocblas-bench\)/\1 -i 1 -j 2/g' ${OUT_DIR}/rocblas_bench.csv > /tmp/rocblas_bench.csv
sed -n '57,$p' /tmp/rocblas_bench.csv > /tmp/rocblas_bench_trail.csv
sh /tmp/rocblas_bench_trail.csv | tee /tmp/rocblas_bench_res.txt
#sed -E -n '/(^N,|^T,)/p' /tmp/rocblas_bench_res.txt > ${OUT_DIR}/rocblas_bench_res.txt

#python3.6 run_rocblas_bench.py -f /tmp/rocblas_bench_trail.csv -o ${OUT_DIR}
