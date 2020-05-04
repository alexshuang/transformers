#!/bin/sh

TOP_DIR=../
EXAMPLE=$TOP_DIR/examples/run_pretraining.py
BASENAME=${EXAMPLE##*/}
FNAME=${BASENAME%.*}
DATA_DIR=data/pretrain/
MODEL_DIR=$DATA_DIR/models
OUT_DIR=$DATA_DIR/out
TRAIN_FILE=$DATA_DIR/wikitext-2-raw/wiki.train.raw
VALID_FILE=$DATA_DIR/wikitext-2-raw/wiki.valid.raw

mkdir -p $OUT_DIR

if [ ! -f $TRAIN_FILE ]; then
	wget -P $DATA_DIR https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
	unzip $DATA_DIR/wikitext-2-raw-v1.zip -d $DATA_DIR
fi

python3.6 $EXAMPLE \
	--output_dir=output \
	--model_type="bert" \
	--model_name_or_path="bert-large-uncased-whole-word-masking" \
	--do_train \
	--train_data_file=$TRAIN_FILE \
	--do_eval \
	--mlm \
	--eval_data_file=$VALID_FILE \
	| tee $DATA_DIR/$FNAME.log
  #--one_iteration \
  #--no_bwd | tee $DATA_DIR/$FNAME.log
