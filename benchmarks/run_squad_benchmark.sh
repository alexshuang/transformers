#!/bin/sh

TOP_DIR=../
EXAMPLE=$TOP_DIR/examples/run_squad.py
BASENAME=${EXAMPLE##*/}
FNAME=${BASENAME%.*}
SQUAD_DIR=data/squad
MODEL_DIR=$SQUAD_DIR/models
OUT_DIR=$SQUAD_DIR/out
#MODEL_NAME=bert-large-uncased
MODEL_NAME=bert-base-uncased
TRAIN_FILE=train-v1.1.json
VALID_FILE=dev-v1.1.json

mkdir -p $OUT_DIR

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

set -e
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${MODEL}_rocblas_bench.csv

/opt/rocm/bin/rocprof -i input.txt --timestamp on --stats -o ${MODEL_NAME}_training_gpu_res.csv \
python3.6 $EXAMPLE \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $SQUAD_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $TRAIN_FILE \
  --predict_file $VALID_FILE \
  --per_gpu_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --one_iter
