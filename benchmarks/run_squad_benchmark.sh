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

OUT_DIR=$SQUAD_DIR/${MODEL_NAME}-seq_len=${SEQ_LEN}-bs=${BS}-steps=${STEPS}

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

#set -e
export ROCBLAS_LAYER=2
export ELAPSED=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
export ROCBLAS_LOG_PROFILE_PATH=${OUT_DIR}/rocblas_profile.csv
export ROCBLAS_LOG_TRACE_PATH=${OUT_DIR}/rocblas_trace.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}

#/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${OUT_DIR}/kernels.csv \
python3.6 $EXAMPLE \
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
  --resoult_dir $OUT_DIR \
  --fp16 \
  --max_steps $STEPS

unset ROCBLAS_LAYER
TOOL=~/rocblas/build/release/clients/staging/rocblas-bench
ln -s ${TOOL} .

#sed 's/\(rocblas-bench\)/\1 -i 1 -j 2/g' ${OUT_DIR}/rocblas_bench.csv > /tmp/rocblas_bench.csv
#sed -n '57,$p' /tmp/rocblas_bench.csv > /tmp/rocblas_bench_trail.csv


sed -n '1161,$p' $ROCBLAS_LOG_BENCH_PATH > /tmp/rocblas_bench_trail.csv
sh /tmp/rocblas_bench_trail.csv | tee /tmp/rocblas_bench_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rocblas_bench_res.txt > ${OUT_DIR}/rocblas_bench_res.txt

#python3.6 run_rocblas_bench.py -f $ROCBLAS_LOG_BENCH_PATH

rm -f ${OUT_DIR}/*.db
