#!/bin/sh

TOP_DIR=../
EXAMPLE=$TOP_DIR/examples/run_squad.py
BASENAME=${EXAMPLE##*/}
FNAME=${BASENAME%.*}
SQUAD_DIR=/data/squad
MODEL_NAME=bert-large-uncased
TRAIN_FILE=train-v1.1.json
VALID_FILE=dev-v1.1.json

SEQ_LEN=512
BS=4
STEPS=${2:-120}
WARMUP_STEPS=20
if [ $STEPS -le $WARMUP_STEPS ]; then
    WARMUP_STEPS=$(expr $STEPS / 5)
fi

OUT_DIR=$1
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

CMD="python3.6 $EXAMPLE \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $SQUAD_DIR \
  --do_train \
  --do_lower_case \
  --train_file $VALID_FILE \
  --per_gpu_train_batch_size $BS \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length $SEQ_LEN \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --overwrite_output_dir \
  --fp16"

set -e

# end2end perf
$CMD --result_dir $OUT_DIR --max_steps $STEPS

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}

echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${OUT_DIR}/kernel_prof.csv \
$CMD --max_steps 1
rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi

unset ROCBLAS_LAYER
sh $ROCBLAS_LOG_BENCH_PATH | tee /tmp/rb_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
