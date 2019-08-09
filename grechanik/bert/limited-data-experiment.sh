#!/bin/sh

DATE="$(date '+%Y-%m-%d_%H.%M.%S')"
LAYERS=6
FP16="--fp16"
#FP16=""
INIT_EPOCHS=8
TAG="AAA-2"


# Just fine-tune on the entire dataset
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
      --eval_with_known_intents \
      --output_dir=$HOME/output/$TAG-ALLINTENTS-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
      --do_train

# Fine-tune on 10 PlayMusic sentences
for i in `seq 0 10 40`; do
      python run_bert_nlu_pytorch.py \
            --bert_model bert-base-uncased \
            --max_seq_length=48 \
            --train_batch_size=32 \
            --learning_rate=2e-5 \
            --num_train_epochs=8.0 \
            --save_checkpoints_steps=5000 \
            --eval_steps=50 \
            --dev_set=0.25 \
            --layers=$LAYERS \
            $FP16 \
            --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                         SearchCreativeWork 0 SearchScreeningEvent 0 PlayMusic 10*150  \
            --shift_limited_data $i \
            --eval_with_known_intents \
            --output_dir=$HOME/output/$TAG-10PM-$i-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
            --do_train
done

# Fine-tune on 6 intents
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
      --eval_with_known_intents \
      --output_dir=$HOME/output/$TAG-ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
      --do_train

# Fine-tune the bert fine-tuned on 6 intents on 10 sentences
for i in `seq 0 10 40`; do
      python run_bert_nlu_pytorch.py \
            --bert_model $HOME/output/$TAG-ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16/checkpoint-final* \
            --max_seq_length=48 \
            --train_batch_size=32 \
            --learning_rate=2e-5 \
            --num_train_epochs=8.0 \
            --save_checkpoints_steps=5000 \
            --eval_steps=50 \
            --dev_set=0.25 \
            --layers=$LAYERS \
            $FP16 \
            --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                         SearchCreativeWork 0 SearchScreeningEvent 0 PlayMusic 10*150  \
            --shift_limited_data $i \
            --eval_with_known_intents \
            --output_dir=$HOME/output/$TAG-10PM-$i-afterONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
            --do_train
done

# Fine-tune only the head of the bert fine-tuned on 6 intents on 10 sentences
for i in `seq 0 10 40`; do
      python run_bert_nlu_pytorch.py \
            --bert_model $HOME/output/$TAG-ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16/checkpoint-final* \
            --max_seq_length=48 \
            --train_batch_size=32 \
            --learning_rate=2e-5 \
            --num_train_epochs=8.0 \
            --save_checkpoints_steps=5000 \
            --eval_steps=50 \
            --dev_set=0.25 \
            --layers=$LAYERS \
            --train_layers_from=100 \
            $FP16 \
            --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                         SearchCreativeWork 0 SearchScreeningEvent 0 PlayMusic 10*150  \
            --shift_limited_data $i \
            --eval_with_known_intents \
            --output_dir=$HOME/output/$TAG-10PM-$i-afterONLY6-FREEZENONHEAD-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
            --do_train
done

# Fine-tune only the head and the last layer of the bert fine-tuned on 6 intents on 10 sentences
for i in `seq 0 10 40`; do
      python run_bert_nlu_pytorch.py \
            --bert_model $HOME/output/$TAG-ONLY6-${INIT_EPOCHS}E-${LAYERS}L$FP16/checkpoint-final* \
            --max_seq_length=48 \
            --train_batch_size=32 \
            --learning_rate=2e-5 \
            --num_train_epochs=8.0 \
            --save_checkpoints_steps=5000 \
            --eval_steps=50 \
            --dev_set=0.25 \
            --layers=$LAYERS \
            --train_layers_from=5 \
            $FP16 \
            --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                         SearchCreativeWork 0 SearchScreeningEvent 0 PlayMusic 10*150  \
            --shift_limited_data $i \
            --eval_with_known_intents \
            --output_dir=$HOME/output/$TAG-10PM-$i-afterONLY6-FREEZE5-${INIT_EPOCHS}E-${LAYERS}L$FP16 \
            --do_train
done
