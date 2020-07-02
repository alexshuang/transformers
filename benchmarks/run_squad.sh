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

MODE=${1:-0} # 0: end2end 1: end2end+rocblas_bench+rocprof
STEPS=${2:-120}
WARMUP_STEPS=20
if [ $STEPS -le $WARMUP_STEPS ]; then
    WARMUP_STEPS=$(expr $STEPS / 5)
fi
NUM_ENTRIES=580

OUT_DIR=$SQUAD_DIR/${MODEL_NAME}-seq_len=${SEQ_LEN}-bs=${BS}-steps=${STEPS}
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
  --fp16 \
  --max_steps $STEPS"

set -e

if [ $MODE -eq 0 ]; then
    $CMD --result_dir $OUT_DIR
elif [ $MODE -ge 1 ]; then
	export ROCBLAS_LAYER=2
	export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
	export ROCBLAS_LOG_PROFILE_PATH=${OUT_DIR}/rocblas_profile.csv
	export ROCBLAS_LOG_TRACE_PATH=${OUT_DIR}/rocblas_trace.csv
	rm -f ${ROCBLAS_LOG_BENCH_PATH}

	echo "pmc: FetchSize L2CacheHit" > input.txt
	/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${OUT_DIR}/model_prof.csv \
	$CMD
	rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt
fi

