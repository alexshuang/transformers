#!/bin/sh

OUT_DIR=${1:-out}
TMP_DIR=$OUT_DIR/tmp
SQUAD_DIR=/data/squad
MODEL_NAME=bert-large-uncased
TRAIN_FILE=train-v1.1.json
VALID_FILE=dev-v1.1.json

STEPS=${2:-1000}
WARMUP_STEPS=30
BS=${3:-4}
SEQ_LEN=${4:-512}

#rm -rf $OUT_DIR
mkdir -p $OUT_DIR $TMP_DIR

if [ ! -f $SQUAD_DIR/$TRAIN_FILE ]; then
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

CMD="python3.6 ../examples/run_squad.py \
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
  --output_dir /tmp/squad \
  --overwrite_output_dir \
  --fp16"

set -e

# end2end perf
$CMD --result_dir $OUT_DIR --max_steps $STEPS | tee $TMP_DIR/run.log

# record kernels
export ROCBLAS_LAYER=6
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
export ROCBLAS_LOG_PROFILE_PATH=${OUT_DIR}/rocblas_config.json
rm -f ${ROCBLAS_LOG_BENCH_PATH}
rm -f ${ROCBLAS_LOG_PROFILE_PATH}
echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${TMP_DIR}/kernel_prof.csv \
$CMD --max_steps 1

sed "s/$/ -i ${STEPS} -j ${WARMUP_STEPS}\/nsleep 1/g" $ROCBLAS_LOG_BENCH_PATH > ${TMP_DIR}/rb.csv
sed -n '/Cijk_A/p' ${TMP_DIR}/kernel_prof.csv > ${OUT_DIR}/kernel_prof.csv

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi
unset ROCBLAS_LAYER
sh ${TMP_DIR}/rb.csv 2>&1 > $TMP_DIR/rb_res.txt | tee $TMP_DIR/rocblas_bench.log
sed -E -n '/(^N,|^T,)/p' $TMP_DIR/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
