# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
sys.path.append('..')
import logging
import argparse
import random
from tqdm import tqdm, trange
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
import pytorch_pretrained_bert.tokenization as tokenization
from pytorch_pretrained_bert.modeling import BertForMultipleChoice_MT_general
from pytorch_pretrained_bert.optimization import BertAdam

import json

from utils_glue import (compute_metrics, processors, GLUE_TASKS_NUM_LABELS, MAX_SEQ_LENGTHS, output_modes)

reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, n_class, do_lower_case,
                                 output_mode, is_multi_choice=True):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    if is_multi_choice:
        features = [[]]
    else:
        features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a.lower() if do_lower_case else example.text_a)  # dialogues

        tokens_b = None

        tokens_c = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b.lower() if do_lower_case else example.text_b)  # answers

        if example.text_c:
            tokens_c = tokenizer.tokenize(example.text_c.lower() if do_lower_case else example.text_c)  # questions

        if tokens_c:
            _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens_b = tokens_c + ["[SEP]"] + tokens_b
        elif tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = (len(tokens_a) + 2) * [0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        pad_length = max_seq_length - len(input_ids)
        input_ids += [0] * pad_length
        input_mask += [0] * pad_length
        segment_ids += [0] * pad_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode in ["classification", "multi-choice"]:
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if is_multi_choice:
            features[-1].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))
            if len(features[-1]) == n_class:
                features.append([])
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))

    if is_multi_choice:
        if len(features[-1]) == 0:
            features = features[:-1]
    print('#features', len(features))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_datasets, model, tokenizer):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    args.train_batch_size = [per_gpu_train_batch_size * max(1, args.n_gpu)
                             for per_gpu_train_batch_size in args.per_gpu_train_batch_size]
    train_iters = []
    tr_batches = []
    for idx, train_dataset in enumerate(train_datasets):
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size[idx])
        train_iters.append(InfiniteDataLoader(train_dataloader))
        tr_batches.append(len(train_dataloader))

    ## set sampling proportion
    total_n_tr_batches = sum(tr_batches)
    sampling_prob = [float(n_batches) / total_n_tr_batches for n_batches in tr_batches]

    t_total = total_n_tr_batches // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         max_grad_norm=args.max_grad_norm,
                         t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %s", ','.join(map(str, args.per_gpu_train_batch_size)))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size[0] * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    nb_tr_examples = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(trange(total_n_tr_batches), desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, _ in enumerate(epoch_iterator):
            epoch_iterator.set_description("train loss: {}".format(tr_loss / nb_tr_examples if nb_tr_examples else tr_loss))
            model.train()

            # select task id
            task_id = np.argmax(np.random.multinomial(1, sampling_prob))
            batch = train_iters[task_id].get_next()

            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3],
                      'task_id':        task_id}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += inputs['input_ids'].size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        if args.do_epoch_checkpoint:
            epoch_output_dir = os.path.join(args.output_dir, 'epoch_{}'.format(epoch))
            os.makedirs(epoch_output_dir, exist_ok=True)
            output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(epoch_output_dir)

        evaluate(args, model, tokenizer, epoch=epoch, is_test=False)
        evaluate(args, model, tokenizer, epoch=epoch, is_test=True)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, epoch=0, is_test=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = args.task_name
    eval_output_dir = args.output_dir

    set_type = 'test' if is_test else 'dev'
    results = {}
    for task_id, eval_task in enumerate(eval_task_names):
        if is_test and not hasattr(processors[eval_task], 'get_test_examples'):
            continue

        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, set_type)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation for {} on {} for epoch {} *****".format(eval_task, set_type, epoch))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        logits_all = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],  # XLM don't use segment_ids
                          'labels':         batch[3],
                          'task_id':        task_id}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                # input_ids, input_mask, segment_ids, label_ids = batch
                # tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, task_id=task_id)

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if logits_all is None:
                logits_all = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                logits_all = np.append(logits_all, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        output_mode = output_modes[eval_task]
        if output_mode in ["classification", "multi-choice"]:
            preds = np.argmax(logits_all, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(logits_all)
        result = compute_metrics(eval_task, preds, out_label_ids.reshape(-1))
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results_{}_{}.txt".format(eval_task, set_type))
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results for {} on {} for epoch {} *****".format(eval_task, set_type, epoch))
            writer.write("***** Eval results for epoch {} *****\n".format(epoch))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            logger.info("\n")

        # get error idx
        correct_idx = np.argwhere(preds == out_label_ids).tolist()
        wrong_idx = np.argwhere(preds != out_label_ids).tolist()
        wrong_idx_dict = {'correct': correct_idx, 'wrong': wrong_idx,
                          'preds': preds.tolist(), 'logits': logits_all.tolist(),
                          'labels': out_label_ids.tolist()}
        json.dump(wrong_idx_dict, open(os.path.join(eval_output_dir,
                                                    "error_idx_{}_{}.json".format(eval_task, set_type)), 'w'))

    return results


def convert_features_to_tensors(features, output_mode, is_multi_choice=True):

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    if is_multi_choice:
        n_class = len(features[0])
        for f in features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)

            label_id.append([f[0].label_id])
    else:
        for f in features:
            input_ids.append(f.input_ids)
            input_mask.append(f.input_mask)
            segment_ids.append(f.segment_ids)
            label_id.append(f.label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    if output_mode in ["classification", "multi-choice"]:
        all_label_ids = torch.tensor(label_id, dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(label_id, dtype=torch.float)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return data


def load_and_cache_examples(args, task, tokenizer, set_type='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    is_multi_choice = True if output_mode == 'multi-choice' else False
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir[task], 'cached_{}_{}_{}_{}'.format(
        set_type,
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(MAX_SEQ_LENGTHS[task]),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir[task])
        label_list = processor.get_labels()
        if set_type == 'train':
            examples = processor.get_train_examples(args.data_dir[task])
        elif set_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir[task])
        else:
            examples = processor.get_test_examples(args.data_dir[task])
        features = convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTHS[task],
                                                tokenizer, len(label_list),
                                                output_mode=output_mode,
                                                do_lower_case=args.do_lower_case,
                                                is_multi_choice=is_multi_choice)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    dataset = convert_features_to_tensors(features, output_mode, is_multi_choice=is_multi_choice)

    return dataset


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir for all tasks, separated by comma ',' ")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # parser.add_argument("--max_seq_length",
    #                     default='128',
    #                     type=str,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default='3', type=str,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="l2 regularization.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--do_epoch_checkpoint",
                        default=False,
                        action='store_true',
                        help="Save checkpoint at every epoch")
    parser.add_argument("--no_cuda",
                        default=False,
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
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.do_train:
            print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args)

    ## prepare tasks
    args.task_name = args.task_name.lower().split(',')
    args.per_gpu_train_batch_size = list(map(int, args.per_gpu_train_batch_size.split(',')))
    for task_name in args.task_name:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))
    args.data_dir = {task_name: data_dir_ for task_name, data_dir_ in zip(args.task_name, args.data_dir.split(','))}
    num_labels = [GLUE_TASKS_NUM_LABELS[task_name] for task_name in args.task_name]
    task_output_config = [(output_modes[task_name], num_label)
                          for task_name, num_label in zip(args.task_name, num_labels)]

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    tokenizer = tokenization.BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertForMultipleChoice_MT_general.from_pretrained(args.bert_model, task_output_config=task_output_config)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train_datasets = [load_and_cache_examples(args, task_name, tokenizer, set_type='train')
                          for task_name in args.task_name]
        global_step, tr_loss = train(args, train_datasets, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # final save of model parameters
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval and not args.do_train:
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))

        model.eval()
        epoch = args.num_train_epochs
        evaluate(args, model, tokenizer, epoch=epoch, is_test=False)
        evaluate(args, model, tokenizer, epoch=epoch, is_test=True)

if __name__ == "__main__":
    main()
