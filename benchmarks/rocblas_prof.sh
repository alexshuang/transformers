#!/bin/sh

set -e

if [ $# -le 0 ]; then
	echo "Usage: ./rocblas_prof.sh <example.py>"
	exit 1
fi

FNAME=${1##*/}
BASENAME=${FNAME%.*}
LOG_DIR=out/$BASENAME

mkdir -p $LOG_DIR

export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=$LOG_DIR/result.csv

SUFFIX=${1##*.}
if [ $SUFFIX = py ]; then
	CMD=python3.6
elif [ $SUFFIX = sh ]; then
	CMD=sh
fi

$CMD $1
echo "rocBLAS logs:" $ROCBLAS_LOG_BENCH_PATH
