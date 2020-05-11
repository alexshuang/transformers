#!/bin/sh

TOP_DIR=../
EXAMPLE=$TOP_DIR/examples/run_squad.py
BASENAME=${EXAMPLE##*/}
FNAME=${BASENAME%.*}
SQUAD_DIR=data/squad
MODEL_DIR=$SQUAD_DIR/models
OUT_DIR=$SQUAD_DIR/out

TRAIN_FILE=train-v1.1.json
VALID_FILE=dev-v1.1.json

mkdir -p $OUT_DIR

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi
#  --cache_dir $MODEL_DIR \

python3.6 $EXAMPLE \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --data_dir $SQUAD_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $TRAIN_FILE \
  --predict_file $VALID_FILE \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --output_dir $OUT_DIR \
  --overwrite_output_dir \
  --one_iteration --no_optim | tee $SQUAD_DIR/$FNAME.log
  #--one_iteration | tee $SQUAD_DIR/$FNAME.log
  #--one_iteration --no_bwd | tee $SQUAD_DIR/$FNAME.log
