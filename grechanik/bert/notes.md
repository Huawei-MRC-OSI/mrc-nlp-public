

## Fine-tuning BERT on the NLU task

* To reproduce run `limited-data-experiment.sh`.
  Tried to randomize the 10 sentences that are used for training. Fine-tuning BERT alone with these
  10 sentences results in 9-23% accuracy (on one intent). Fine-tuning on 10 sentences after
  fine-tuning on the other 6 intents results in 15-30% accuracy (on one intent), but at the cost of
  forgetting all 6 intents (they are not completely forgotten, accuracy for all 7 intents drops from
  76% to 30-55% depending on the 10 sentences).
  If we finetune only the head on these 10 sentences, we get 7-12% accuracy but without any drop on
  the other intents.
  If we finetune the head and the last layer, we get 18-30% (more like around 21% though).
  If we finetune only the head without prefinetuning on 6 intents, we get 14%.

* Changed the head for bert. Now every intent has a separate dense layer for classifying slots, and
  we train only one head on each example.

* Tried finetuning on 6 intents and then finetuning only on 10 phrases from a single intent. The
  result depends on the number of epochs of the first 6-intent finetuning. For 0 epochs we gen about
  32-36% wfa on PlayMusic, for 1-15 epochs we get 50-55%. However on 16 epochs it breaks and we get
  0%.

* Added distiller to the bert script. It works but it seems like removing whole layers is better
  than pruning.

* How to use distiller in your script is described quite well in this tutorial:
  <https://nervanasystems.github.io/distiller/tutorial-lang_model.html>

* And another experiment. Can we train BERT just on single intent? Yes, we can, I reached 84% wfa
  on PlayMusic by training just on this intent (training together with other intents results in
  80-85% on PlayMusic).
  ```
  python run_bert_nlu_pytorch.py \
      --bert_model=bert-base-uncased \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=30.0 \
      --save_checkpoints_steps=500 \
      --eval_steps=250 \
      --eval_on_test --dev_set=0 \
      --layers=6 \
      --fp16 \
      --limit_data AddToPlaylist 0 BookRestaurant 0 GetWeather 0 RateBook 0 \
                   SearchCreativeWork 0 SearchScreeningEvent 0  \
      --output_dir=$HOME/output/pytorch-bert-nlu-PMALL-OTHER0-L6-FP16-$(date '+%Y-%m-%d_%H.%M.%S') \
    "$@"
  ```

* Another experiment. Let's assume that we have only 100 examples of PlayMusic and we can train only
  the last layer with it (together with the other data though). And we can perform pretraining for
  the whole bert on the rest of the data.
  - First train the last layer without pretraining. This achieves 80%wfa and 50%wfa-on-PlayMusic
    ```
    python run_bert_nlu_pytorch.py \
      --bert_model=bert-base-uncased \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=30.0 \
      --save_checkpoints_steps=500 \
      --eval_steps=250 \
      --eval_on_test --dev_set=0 \
      --layers=6 \
      --fp16 \
      --train_layers_from 5 \
      --limit_data PlayMusic 100 \
      --output_dir=$HOME/output/pytorch-bert-nlu-PM100-TRLAST-L6-FP16-$(date '+%Y-%m-%d_%H.%M.%S') \
    "$@"
    ```
  - Now pretrain on data without PlayMusic, but all layers. This achieves 77%wfa with 0% on
    PlayMusic obviously
    ```
    python run_bert_nlu_pytorch.py \
      --bert_model=bert-base-uncased \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=30.0 \
      --save_checkpoints_steps=500 \
      --eval_steps=250 \
      --eval_on_test --dev_set=0 \
      --layers=6 \
      --fp16 \
      --limit_data PlayMusic 0 \
      --output_dir=$HOME/output/pytorch-bert-nlu-PM0-PRETRAIN-L6-FP16-$(date '+%Y-%m-%d_%H.%M.%S') \
      "$@"
    ```
  - Now train the last layer of the pretrained on the whole data. This achieves 85%wfa and 52%wfa on
    PlayMusic. This is just slightly better for PlayMusic, however other intents benefited
    considerably from whole-model pretraining. (we haven't reached 60% on PlayMusic which is
    achievable with whole-model fine-tuning)
    ```
    python run_bert_nlu_pytorch.py \
      --bert_model=/workspace/output/pytorch-bert-nlu-PM0-PRETRAIN-L6-FP16-2019-06-27_15.38.54/checkpoint-11070 \
      --max_seq_length=48 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=30.0 \
      --save_checkpoints_steps=500 \
      --eval_steps=250 \
      --eval_on_test --dev_set=0 \
      --layers=6 \
      --fp16 \
      --train_layers_from 5 \
      --limit_data PlayMusic 100 \
      --output_dir=$HOME/output/pytorch-bert-nlu-PM100-TRLAST-AFTER-PRETRAIN-L6-FP16-$(date '+%Y-%m-%d_%H.%M.%S') "$@"
    ```

* Conducted some simple experiments with limited data: I limit data for some intent to a certain
  number of examples and train like this. First of all, it is very important to resample so that the
  training data remains balanced (without it the network will converge much slower).
  For PlayMusic
  Original 2000 examples get us to 80-85% whole frame accuracy
  500 examples to about 77%
  200 examples to 63-68%
  100 examples to 55-60%
  10 examples to 10-20%
  ```
  python run_bert_nlu_pytorch.py \
    --bert_model=bert-base-uncased \
    --max_seq_length=48 \
    --train_batch_size=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=30.0 \
    --save_checkpoints_steps=500 \
    --eval_steps=250 \
    --eval_on_test --dev_set=0 \
    --layers=6 \
    --fp16 \
    --limit_data PlayMusic 200 \
    --output_dir=$HOME/output/pytorch-bert-nlu-PM200-L6-FP16-$(date '+%Y-%m-%d_%H.%M.%S') \
    "$@"
  ```

* Training just the top of the network (dense layers) doesn't work (it's quite understandable why).
  Adding just one last transformer layer works quite well, about 83% whole frame accuracy. Training
  2 transformer layer brings this number to 87%, and training 3 layers gets us to 89% (I
  experimented with 6-layer bert and fp16).

* Turns out that separating intents and B-/I- classification doesn't help. Moreover, separating
  intents is bad, because it increases importance of intents during training (a fix is to divide the
  intent loss by e.g. 100).

* Found a bug in intent accuracy computation, so BERT doesn't achieve 99.7%, just 99%.

* Tried bert-large with fp16, it is about 3x slower than bert-base with fp16 on both training and
  evaluation. Doesn't train faster, doesn't achieve better scores. For our task it is useless.
  bert-large with fp16 requires about 7.7GB during training. The fp32 version needs 9.7GB. (This is
  pytorch).

* FP16 training works, almost 2x faster, but consumes about 30% less memory.

* Pruning by masking channels with the lowest abs value of mask derivatives does work. There is no
  noticeable drop in accuracy when 20% of channels are removed, and I suppose we can remove more
  (and I do experiments with the 6-layer BERT, so it's already faster than BASE). However I have no
  idea how much we gain in terms of efficiency, because I don't physically remove the weights.
  (See `~grechanik/docker-home/output/pytorch-bert-nlu-prunetest-6l-2019-06-20_15.15.19`).

* For some reason the ordinary Adam optimizer (not BertAdam) doesn't work with BERT (it unlearns
  instead of learning). I don't know the reason, may be due to different weight decay strategy.

* Redid experiments with reduced number of layers. Interestingly 6-layer bert is almost as good as
  12-layer bert.

* Some things we may try in order to improve results on NLU:
  - Better output formulation (separate slot and intention labels, separate output for B-/I-).
  - Finding better learning rate, trying cyclic lr, trying some superconvergence tricks.
  - Data augmentation (e.g. replacing slots with same-labeled slots from other sentences).
  - Tune dropout and weight decay parameters.
  - Focal loss.

* Tried the pytorch version of BERT. It is as good as the tf version on NLU. Maybe pytorch is a bit
  faster, also it consumes less memory by default (I guess we can make tf consume less memory too by
  explicitly limiting it but I haven't tried).

* Note that if you get zero whole frame accuracy, you might have forgotten to take input masks into
  account (if you use masks for computing the loss then you probably must use masks to compute the
  whole frame accuracy). This was fixed in this commit.

* Tried to play with weighted losses. Didn't work, all weighting approaches I tried seemed harmful
  or useless. Note that I may have made some mistake, so I would encourage you to try it yourself
  (in particular I mix intents and slots). One important thing is that you might want to scale the
  loss or the learning rate.
  Also tried to divide the sentence loss by the sentence length, seems a little bit harmful.

* Tensorflow profiling. General instructions are
  [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md),
  you need to build the profile-reading tool with bazel.
  ```
  # You may want to say NO to XLA during configuration (to prevent downloading llvm which may fail
  #  to download)
  ./configure
  bazel build --config=opt tensorflow/core/profiler:profiler

  bazel-bin/tensorflow/core/profiler/profiler \
    --profile_path=/tmp/train_dir/profile_xx

  # Within the tool the most useful command looks like this:
  scope -max_depth 30 -select micros -min_micros 100 -order_by micros
  ```
  There is also some kind of gui but I was unable to make it work.
  And it seems most of the time in BERT is taken by matrix multiplication. (LayerNorm is also quite
  time-consuming.)

* Turned out, all this time I trained on the dev set and evaluated on the train set. Now I fixed the
  bug and we magically have more training data, so the results are better, namely
  90% whole frame accuracy (was 85%) (With a different measuring method frame acc is 92%).

* It is possible to use hidden layers of BERT as outputs. Speed will be higher, accuracy will be
  worse. Just two layers reach approx 70% frame acc, 6 layers approx 82%, 8 layers ~84%, and
  the full 12-layer base bert reaches ~85% (WARNING: this is on the smaller train set!).

* The intent of "please play kabhi jo baadal barse by ruth lorenzo" is PlayMusic, but the intent of
  "play the song red lanta" is SearchCreativeWork. WTF?

* I modified metrics computation: now I convert the list of predicted slot labels into a more
  coarser list (for original sentence pieces instead of for tokens). This improves some scores, like
  whole frame acc goes from 85% to 88%. However I'm sure this is NOT the right thing to do, because
  this basically leaks the information about entity boundaries which is not available in real life.
  I guess the right thing to do is to perform two-step tokenization: first split into words, and
  then split into BERT tokens. Metrics should be computed on the level of words (not implemented
  yet).

* Don't forget about masks! I forgot about masks and have been getting too optimistic accuracies
  for slot filling.

* Tried training BERT without pretraining. Only 51% whole frame accuracy in 20 epochs.

* The metric which I call success rate seems to be called "whole frame accuracy" in the literature.

* Using the f1 score computation function from sklearn, I get the following results for bert:
  - micro and weighted f1 scores are about 0.99
  - macro f1 is about 0.93 which may be due to class imbalance.

* Be careful with the f1 score! It shouldn't be computed batchwise if the metrics are averaged
  across all batches. Also in my implementation I split original sentence pieces into smaller
  tokens, so the f1 score may also be different.

* TF Estimators somewhat support validation during training, however:
  - Validation is run on a separate graph which is built every time we want to validate.
  - Weights are loaded from the last checkpoint, so you cannot validate more frequently than making
    checkpoints.

* Fine-tuning BERT requires about 8.5 GB with batch size 32. The speed is about 4.5 steps/sec. The
  validation loss is minimized around 600-th step (approx 20000 examples), then it starts
  overfitting. However accuracy and success rate still grow after this point.

* Since we don't compute the f1-score yet, I propose using another metric which I call success rate
  (seems like it is usually called whole frame accuracy): the fraction of perfectly labelled
  sentences. With BERT I get only about 85% on my validation set. Flags to reproduce:
  ```
  export BERT_BASE_DIR="$HOME/proj/mrc-nlp/bert-google/uncased_L-12_H-768_A-12"

  python run_bert_nlu.py \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=48 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=20.0 \
    --save_checkpoints_steps=200 \
    --save_summary_steps=100 \
    --output_dir=$HOME/output/bert-nlu-$(date '+%Y-%m-%d_%H.%M.%S') \
    --run_train --run_eval
  ```

