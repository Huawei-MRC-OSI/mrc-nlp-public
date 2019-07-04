# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner.

python run_bert_nlu_pytorch.py \
  --bert_model=bert-base-uncased \
  --max_seq_length=48 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=20.0 \
  --save_checkpoints_steps=500 \
  --eval_steps=250 \
  --eval_on_test --dev_set=0 \
  --output_dir=$HOME/output/pytorch-bert-nlu-TEST-$(date '+%Y-%m-%d_%H.%M.%S') \
  "$@"

The model trained on 20 epochs is here:
~grechanik/docker-home/output/pytorch-bert-nlu-TEST-2019-06-11_15.23.27

The results are:
intent_accuracy  99.7
f1_score_seqeval 95.2
whole_frame_accuracy 88.8

Slightly worse than the tf version, the reason is unknown.

"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
import time
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
import sklearn.metrics
from seqeval.metrics.sequence_labeling import f1_score as f1_score_seqeval

from my_pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from my_pytorch_pretrained_bert.modeling import BertForNLU, BertConfig, load_tf_weights_in_bert, PrunableLinear
from my_pytorch_pretrained_bert.tokenization import BertTokenizer
from my_pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)

########################################################################################
# https://stackoverflow.com/questions/15445981/how-do-i-disable-the-security-certificate-check-in-python-requests

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False

    return settings

requests.Session.merge_environment_settings = merge_environment_settings

########################################################################################

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default=os.path.expanduser("~/proj/nlu-benchmark/"),
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--from_tf",
                    action='store_true',
                    help="Consider bert_model to be a tf checkpoint")
parser.add_argument("--cache_dir",
                    default=os.path.expanduser("~/cache/"),
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--eval_on_test",
                    action='store_true',
                    help="Whether to use the test set instead of the dev set.")
parser.add_argument("--do_lower_case",
                    default=True,
                    type=bool,
                    help="Whether to use uncased model.")
parser.add_argument("--layers",
                    default=None,
                    type=int,
                    help="Use this number of layers")
parser.add_argument("--train_layers_from",
                    default=None,
                    type=int,
                    help="Train only top layers, starting from this one.")
parser.add_argument("--dev_set",
                    default=0.25,
                    type=float,
                    help="The portion of train data to use as a dev set.")
parser.add_argument("--limit_data",
                    default=None,
                    type=str,
                    nargs='+',
                    help="Limit data for the specified intents.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--const_lr",
                    action='store_true',
                    help="Whether to simply use const lr schedule.")
parser.add_argument("--dropout",
                    default=None,
                    type=float,
                    help="Override BERT dropout settings.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument("--distiller",
                    type=str,
                    default=None,
                    help="A distiller conf file to use for pruning with distiller.")
parser.add_argument("--prune",
                    action='store_true',
                    help="Perform pruning after each evaluation (experimental).")
parser.add_argument("--prune_count",
                    default=500,
                    type=int,
                    help="The number of channels to prune each time")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument("--save_checkpoints_steps",
                    default=500,
                    type=int)
parser.add_argument("--eval_steps",
                    default=250,
                    type=int)
args = parser.parse_args()

if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, *,
                 input_tokenids, # [101, 123, 456, ...], ids of WordPiece tokens, padded and with cls
                 input_mask,     # [1, 1, 1, ..., 1, 0, 0, 0, ...], including cls
                 input_segmentids, # Just zeros
                 input_labelids, # [intent_id, slot_ids, ...]
                 input_tokens, # ['[CLS]', 'a', '##b', ..., '[SEP]', 0, 0, ...]
                 input_word_indices, # [-1, 0, 0, 1, 2, 2, ...], indices of words in wordinput
                                     #                           for each token (including cls)
                 wordinput_labelids, # label ids for word representation, without intent/cls
                 wordinput, # ['ab', 'the', 'one', ...], word representation, without intent/cls
                 is_real_example=True):
        self.input_tokenids = input_tokenids
        self.input_mask = input_mask
        self.input_segmentids = input_segmentids
        self.input_labelids = input_labelids
        self.input_tokens = input_tokens
        self.is_real_example = is_real_example
        self.input_word_indices = input_word_indices
        self.wordinput_labelids = wordinput_labelids
        self.wordinput = wordinput

    def untokenize_prediction(self, prediction):
        """Convert a list of labels for a fine-grained sentence representation into a list for
        the original coarser representation. This functions assumes that prediction does NOT contain
        the intent (cls token)."""
        res = [0]*len(self.wordinput_labelids)
        for i, l in zip(self.input_word_indices[1:], prediction):
            if i != -1:
                if res[i] == 0:
                    res[i] = l
        return res


class DataProcessor(object):
    def __init__(self, data_dir):
      pass

    def get_train_examples(self):
      raise NotImplementedError()

    def get_dev_examples(self):
      raise NotImplementedError()

    def get_test_examples(self):
      raise NotImplementedError()

    def get_labels(self):
      raise NotImplementedError()


class NLUDataProcessor(DataProcessor):
    def __init__(self, data_dir, max_seq_length, tokenizer):
        logger.info("Loading data from %s", data_dir)
        intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                   'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

        self.intents = intents
        self.labels_to_ids = {'O': 0}
        self.labels = ['O']
        self.label_counts = [0]
        self.max_seq_length = max_seq_length

        train_data = {}
        for intent in intents:
            f = open(data_dir + '/2017-06-custom-intent-engines/' + intent +
                     '/train_' + intent + '_full.json', encoding='latin_1')
            sentences = json.load(f)[intent]
            f.close()
            for elem in sentences:
                sentence = [x['text'] for x in elem['data']]
                labels = [x['entity'] if 'entity' in x.keys() else 'O' for x in elem['data']]
                train_data.setdefault(intent, [])\
                    .append(self.to_features(tokenizer, sentence, labels, intent))

        test_data = []
        for intent in intents:
            f = open(data_dir + '/2017-06-custom-intent-engines/' + intent +
                     '/validate_' + intent + '.json', encoding='latin_1')
            sentences = json.load(f)[intent]
            f.close()
            for elem in sentences:
                sentence = [x['text'] for x in elem['data']]
                labels = [x['entity'] if 'entity' in x.keys() else 'O' for x in elem['data']]
                test_data.append(self.to_features(tokenizer, sentence, labels, intent, is_test=True))

        # Reset the seed here just to make sure that we shuffle data with the same seed
        random.seed(args.seed)
        random.shuffle(test_data)
        self.test_data = test_data
        for data in train_data.values():
            random.shuffle(data)
        self.train_data = train_data
        logger.info("Done loading data")

    def get_train_examples(self, restrict={}):
        res = []
        for intent, data in self.train_data.items():
            data = data[:int(len(data)*(1 - args.dev_set))]
            if intent in restrict:
                restr = restrict[intent]
                scale = None
                if not isinstance(restr, tuple):
                    restr = (restr,)
                if isinstance(restr, tuple):
                    if len(restr) == 1:
                        restr = restr[0]
                        if restr != 0:
                            scale = (len(data) // restr)
                        else:
                            scale = 0
                    else:
                        restr, scale = restrict[intent]
                data = data[:restr] * scale
            print("test data for intent", intent, "size", len(data))
            res += data
        random.shuffle(res)
        return res

    def get_dev_examples(self):
        res = []
        for intent, data in self.train_data.items():
            data = data[int(len(data)*(1 - args.dev_set)):]
            res += data
        random.shuffle(res)
        return res

    def get_test_examples(self):
        return self.test_data

    def get_labels(self):
        return self.labels

    def to_features(self, tokenizer, data_pieces, label_pieces, intent, is_test=False):
        wordinput = []
        wordinput_labels = []
        input_tokens = ['[CLS]']
        token_labels = [intent]
        input_word_indices = [-1]
        word_index = 0
        for piece, label in zip(data_pieces, label_pieces):
            words = piece.split()
            if label == 'O':
                word_labels = [label]*len(words)
            else:
                word_labels = ["B-" + label] + ["I-" + label]*(len(words) - 1)
            wordinput += words
            wordinput_labels += word_labels

            for d, l in zip(words, word_labels):
                d_tokens = tokenizer.tokenize(d)
                input_tokens += d_tokens
                input_word_indices += [word_index]*len(d_tokens)
                word_index += 1
                token_labels += [l]*len(d_tokens)

        self._add_labels(token_labels[1:])

        if len(input_tokens) >= self.max_seq_length:
            print("Max length exceeded: ", len(input_tokens))

        input_tokens = input_tokens[:self.max_seq_length-1]
        token_labels = token_labels[:self.max_seq_length-1]

        for l in token_labels[1:]:
            self.label_counts[self.labels_to_ids[l]] += 1

        input_tokens += ['[SEP]']
        token_labels += ['O']
        input_word_indices += [-1]

        input_tokenids = tokenizer.convert_tokens_to_ids(input_tokens)
        token_labels = ([self.intents.index(token_labels[0])] +
                        [self.labels_to_ids[l] for l in token_labels[1:]])
        input_mask = [1] * len(input_tokens)

        # Zero-pad up to the sequence length.
        while len(input_tokenids) < self.max_seq_length:
            input_tokenids.append(0)
            input_mask.append(0)
            token_labels.append(0)
            input_word_indices.append(-1)

        wordinput_labelids = [self.labels_to_ids[l] for l in wordinput_labels]

        return InputFeatures(input_tokenids=input_tokenids,
                             input_mask=input_mask,
                             input_segmentids=[0]*self.max_seq_length,
                             input_labelids=token_labels,
                             input_tokens=input_tokens,
                             input_word_indices=input_word_indices,
                             wordinput_labelids=wordinput_labelids,
                             wordinput=wordinput)

    def _add_labels(self, labels):
        for l in labels:
            if l not in self.labels_to_ids:
                self.labels_to_ids[l] = len(self.labels_to_ids)
                self.labels.append(l)
                self.label_counts.append(0)


def flatten(lst):
    return [x for l in lst for x in l]


def main():
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty."
                         .format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    processor = NLUDataProcessor(args.data_dir, args.max_seq_length, tokenizer)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    num_intents = len(processor.intents)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        restrict = {}
        if args.limit_data:
            for i in range(len(args.limit_data) // 2):
                intent = args.limit_data[i*2]
                size = args.limit_data[i*2 + 1]
                assert intent in processor.intents
                restrict[intent] = tuple(int(x) for x in size.split('*'))
        train_examples = processor.get_train_examples(restrict)
        num_train_optimization_steps = \
            int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * \
            args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = \
                num_train_optimization_steps // torch.distributed.get_world_size()

    if args.eval_on_test:
        eval_examples = processor.get_test_examples()
    else:
        eval_examples = processor.get_dev_examples()

    # Prepare model
    cache_dir = args.cache_dir or os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                               'distributed_{}'.format(args.local_rank))
    model = BertForNLU.from_pretrained(args.bert_model,
                                       cache_dir=cache_dir,
                                       num_labels=num_labels,
                                       num_intents=num_intents,
                                       from_tf=args.from_tf,
                                       layers=args.layers,
                                       prune=args.prune,
                                       dropout=args.dropout)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # distiller
    compression_scheduler = None
    if args.distiller is not None:
        import distiller
        distiller_pylogger = distiller.data_loggers.PythonLogger(logger)
        compression_scheduler = distiller.config.file_config(model, None, args.distiller)

    # Tensorboard
    swriter = SummaryWriter(args.output_dir)
    logger.info("Writing summary to %s", args.output_dir)

    # Prepare optimizer
    if args.do_train:
        if args.train_layers_from is not None:
            param_optimizer = []
            for n, p in model.named_parameters():
                if "classifier" in n or "pooler" in n:
                    param_optimizer.append((n, p))
                elif any(int(s) >=args.train_layers_from for s in re.findall(r'layer\.(\d+)\.', n)):
                    param_optimizer.append((n, p))
                else:
                    print("Not considered trainable:", n)
                    p.requires_grad_(False)
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        if args.fp16:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps,
                                 schedule=None if args.const_lr else 'warmup_linear')

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_tokenids = torch.tensor([f.input_tokenids for f in train_examples],
                                          dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_examples], dtype=torch.long)
        all_input_segmentids = torch.tensor([f.input_segmentids for f in train_examples],
                                            dtype=torch.long)
        all_input_labelids = torch.tensor([f.input_labelids for f in train_examples],
                                          dtype=torch.long)

        train_data = TensorDataset(all_input_tokenids, all_input_mask,
                                   all_input_segmentids, all_input_labelids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size)

        batches_per_epoch = int(len(train_examples) / args.train_batch_size)

        model.train()
        for epoch_id in trange(int(args.num_train_epochs), desc="Epoch"):
            if compression_scheduler:
                compression_scheduler.on_epoch_begin(epoch_id)
            nb_tr_examples = 0
            global_step_tr_loss = 0.0
            tqdm_bar = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(tqdm_bar):
                if compression_scheduler and step % args.gradient_accumulation_steps == 0:
                    compression_scheduler.on_minibatch_begin(
                        epoch_id, minibatch_id=step/args.gradient_accumulation_steps,
                        minibatches_per_epoch=batches_per_epoch)

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss, _, _ = model(input_ids, segment_ids, input_mask,
                                   labels=label_ids)

                if compression_scheduler:
                    # Before running the backward phase, we allow the scheduler to modify the loss
                    # (e.g. add regularization loss)
                    loss = compression_scheduler.before_backward_pass(
                        epoch_id, minibatch_id=step/args.gradient_accumulation_steps,
                        minibatches_per_epoch=batches_per_epoch, loss=loss,
                        return_loss_components=False)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                global_step_tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    tqdm_bar.set_postfix(train_loss=global_step_tr_loss)
                    swriter.add_scalar('train_loss',
                                       global_step_tr_loss,
                                       global_step=global_step)
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = \
                            args.learning_rate * warmup_linear.get_lr(global_step,
                                                                      args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    if args.fp16:
                        for _, p in param_optimizer:
                            if p.grad is None:
                                p.grad = torch.zeros(p.size(), dtype=p.dtype, device=p.device)
                    optimizer.step()
                    global_step += 1
                    global_step_tr_loss = 0.0

                    if compression_scheduler:
                        compression_scheduler.on_minibatch_end(
                            epoch_id, minibatch_id=step/args.gradient_accumulation_steps,
                            minibatches_per_epoch=batches_per_epoch)

                    optimizer.zero_grad()

                    if not args.fp16 and optimizer.get_lr():
                        # get_lr returns a list, however all the elements of the list should be the
                        # same
                        swriter.add_scalar('learning_rate',
                                           np.random.choice(optimizer.get_lr()),
                                           global_step=global_step)

                    if global_step % args.eval_steps == 0:
                        perform_evaluation(eval_examples=eval_examples,
                                           model=model,
                                           processor=processor,
                                           swriter=swriter,
                                           device=device,
                                           global_step=global_step)
                        model.train()

                    if global_step % args.save_checkpoints_steps == 0:
                        save_model(model=model, tokenizer=tokenizer, global_step=global_step)

                    if args.prune and global_step % args.eval_steps == 1:
                        prune_model(model=model,
                                    swriter=swriter,
                                    global_step=global_step,
                                    count=args.prune_count)

            if compression_scheduler:
                sparsity_table, total_sparsity = \
                    distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
                logger.info("\nParameters:\n" + str(sparsity_table))
                logger.info('Total sparsity: {:0.2f}\n'.format(total_sparsity))
                swriter.add_scalar('sparsity', total_sparsity, global_step=global_step)
                compression_scheduler.on_epoch_end(epoch_id)

        save_model(model=model, tokenizer=tokenizer, global_step=global_step, tag='final')

    perform_evaluation(eval_examples=eval_examples,
                       model=model,
                       processor=processor,
                       swriter=swriter,
                       device=device,
                       global_step=global_step)
    swriter.close()


def perform_evaluation(eval_examples, model, processor, swriter, device, global_step=None):
    output_eval_file = os.path.join(args.output_dir,
                                    "eval_results{}.txt"
                                    .format("" if global_step is None else global_step))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_tokenids = torch.tensor([f.input_tokenids for f in eval_examples],
                                          dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_examples], dtype=torch.long)
        all_input_segmentids = torch.tensor([f.input_segmentids for f in eval_examples],
                                            dtype=torch.long)
        all_input_labelids = torch.tensor([f.input_labelids for f in eval_examples],
                                            dtype=torch.long)

        label_list = processor.get_labels()
        num_labels = len(label_list)

        eval_data = TensorDataset(all_input_tokenids, all_input_mask,
                                  all_input_segmentids, all_input_labelids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        time_begin = time.time()
        for input_ids, input_mask, segment_ids, label_ids in \
                tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss, intent_logits, slot_logits = \
                    model(input_ids, segment_ids, input_mask, labels=label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            intent_ids = list(np.argmax(intent_logits.detach().cpu().numpy(), axis=-1))
            slot_ids = list(np.argmax(slot_logits.detach().cpu().numpy(), axis=-1))
            preds += list(zip(intent_ids, slot_ids))

        time_end = time.time()

        eval_result = {'loss': eval_loss/nb_eval_steps}

        num_actual_eval_examples = len(eval_examples)

        # Intent ids from all examples
        all_true_intent_ids = []
        all_pred_intent_ids = []

        # Slot labels for all sentences
        seqeval_true_labels = {intent: [] for intent in processor.intents}
        seqeval_pred_labels = {intent: [] for intent in processor.intents}

        intent_num_examples = {intent: 0 for intent in processor.intents}
        intent_num_fully_correct = {intent: 0 for intent in processor.intents}
        intent_num_intent_correct = {intent: 0 for intent in processor.intents}

        mispredictions_count = 0
        for example, (pred_intent_id, pred_slot_ids) in zip(eval_examples, preds):
            intent = processor.intents[example.input_labelids[0]]
            intent_num_examples[intent] += 1

            full_token_true_ids = example.input_labelids
            full_token_pred_ids = [pred_intent_id] + list(pred_slot_ids)
            wordinput_pred_ids = example.untokenize_prediction(pred_slot_ids)
            wordinput_true_ids = example.wordinput_labelids
            true_label_names = [processor.labels[l] for l in wordinput_true_ids]
            pred_label_names = [processor.labels[l] for l in wordinput_pred_ids]

            all_true_intent_ids.append(example.input_labelids[0])
            all_pred_intent_ids.append(pred_intent_id)

            seqeval_true_labels[intent].append(true_label_names)
            seqeval_pred_labels[intent].append(pred_label_names)

            writer.write("\n")
            writer.write("%s\n" % " ".join(example.input_tokens))
            writer.write("True intent: {} {}\n"
                         .format(example.input_labelids[0], intent))
            writer.write("Pred intent: {} {}\n"
                         .format(pred_intent_id, processor.intents[pred_intent_id]))
            writer.write("True word ids: %s\n" % str(wordinput_true_ids))
            writer.write("Pred word ids: %s\n" % str(wordinput_pred_ids))
            writer.write("%s\n" % list(zip(example.wordinput, true_label_names)))
            writer.write("%s\n" % list(zip(example.wordinput, pred_label_names)))
            for token, correct, predicted in \
                    zip(example.input_tokens, full_token_true_ids, full_token_pred_ids):
                if correct != predicted:
                    writer.write("ERR: %s is %s but predicted %s\n" %
                                 (token, processor.labels[correct], processor.labels[predicted]))

            if (np.array(wordinput_pred_ids) != np.array(wordinput_true_ids)).any() or \
                    pred_intent_id != full_token_true_ids[0]:
                mispredictions_count += 1
                writer.write("CONSIDERED WRONG!\n")
            else:
                intent_num_fully_correct[intent] += 1
                writer.write("CONSIDERED RIGHT!\n")

            if pred_intent_id == full_token_true_ids[0]:
                intent_num_intent_correct[intent] += 1

        eval_result['f1_score_seqeval'] = \
            f1_score_seqeval(flatten(seqeval_true_labels.values()),
                             flatten(seqeval_pred_labels.values()))

        eval_result['whole_frame_accuracy'] = 1 - mispredictions_count / num_actual_eval_examples
        eval_result['intent_accuracy'] = \
            sklearn.metrics.accuracy_score(all_true_intent_ids, all_pred_intent_ids)
        eval_result['eval_time'] = time_end - time_begin

        for intent in processor.intents:
            eval_result['f1_score_seqeval_' + intent] = \
                f1_score_seqeval(seqeval_true_labels[intent],
                                 seqeval_pred_labels[intent])
            eval_result['whole_frame_accuracy_' + intent] = \
                intent_num_fully_correct[intent] / intent_num_examples[intent]
            eval_result['intent_accuracy_' + intent] = \
                intent_num_intent_correct[intent] / intent_num_examples[intent]

        logger.info("***** Eval results *****")
        for key in sorted(eval_result.keys()):
            swriter.add_scalar("eval_" + key, eval_result[key], global_step=global_step)
            logger.info("  %s = %s", key, str(eval_result[key]))
            writer.write("%s = %s\n" % (key, str(eval_result[key])))

        #  with h5py.File(os.path.join(args.output_dir, "internal_values.h5"), 'w') as internal_values_h5:
        #    for name in internal_values:
        #      print("Writing", name)
        #      internal_values_h5.create_dataset(name, data=np.concatenate(internal_values[name]))

        print(output_eval_file)


def prune_model(model, swriter, global_step, count):
    logger.info("Pruning")
    total = 0
    places = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            for mask in [m.in_mask, m.out_mask]:
                total += mask.size()[0]
                if mask.grad is not None:
                    grad = mask.grad.detach().cpu().numpy()
                    for i in np.argwhere(mask.detach().cpu().numpy() != 0.0):
                        places.append((np.abs(grad[i]), mask, i))
                    mask.requires_grad_(False)

    logger.info("Sorting")
    places.sort(key=lambda t: t[0])
    score_vals = np.array([t[0] for t in places])
    if len(score_vals) > 0:
        swriter.add_histogram("pruning_score_vals", score_vals, global_step=global_step)

    logger.info("Assigning")
    for _, mask, i in places[:count]:
        mask[i] = 0.0
    total_unpruned = 0
    logger.info("Computing the number of unpruned")
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            for mask in [m.in_mask, m.out_mask]:
                total_unpruned += int(np.sum(mask.detach().cpu().numpy() != 0.0))
    logger.info("Unpruned %s (%s of %s)", total_unpruned/total, total_unpruned, total)
    swriter.add_scalar("unpruned_fraction", total_unpruned/total, global_step=global_step)

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            for mask in [m.in_mask, m.out_mask]:
                if mask.grad is not None:
                    mask.grad.zero_()
                mask.requires_grad_(True)


def save_model(model, tokenizer, global_step, tag=None):
    model_to_save = model.module if hasattr(model, 'module') else model

    if tag is None:
        filename = "checkpoint-{}".format(global_step)
    else:
        filename = "checkpoint-{}-{}".format(tag, global_step)

    model_dir = os.path.join(args.output_dir, filename)
    os.makedirs(model_dir)
    logger.info("Saving %s", model_dir)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_dir)

if __name__ == "__main__":
    main()
