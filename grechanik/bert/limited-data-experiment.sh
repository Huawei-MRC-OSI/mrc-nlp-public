#!/bin/sh

DATE="$(date '+%Y-%m-%d_%H.%M.%S')"
LAYERS=6
FP16="--fp16"
INIT_EPOCHS=8

python run_bert_nlu_pytorch.py \
      --bert_model=bert-base-uncased \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=$INIT_EPOCHS \
      --save_checkpoints_steps=5000 \
      --eval_steps=250 \
      --dev_set=0.25 \
      --layers=$LAYERS \
      $FP16 \
      --limit_data PlayMusic 0  \
      --output_dir=$HOME/output/ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16-$DATE \
      --do_train

python run_bert_nlu_pytorch.py \
      --bert_model $HOME/output/ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16-$DATE/checkpoint-final* \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=20.0 \
      --save_checkpoints_steps=5000 \
      --eval_steps=50 \
      --dev_set=0.25 \
      --layers=$LAYERS \
      $FP16 \
      --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                   SearchCreativeWork 0 SearchScreeningEvent 0 PlayMusic 10  \
      --output_dir=$HOME/output/10ofPM-after-ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16-$DATE \
      --do_train
