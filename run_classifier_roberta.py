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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import sys
import json
sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.modeling import BertForMultipleChoice_MT_general

from pytorch_transformers import (WEIGHTS_NAME, CONFIG_NAME,
                                  BertConfig, BertTokenizer,
                                    BertForMultipleChoice_MT_general,
                                  # XLMConfig, XLMForSequenceClassification,
                                  # XLMTokenizer,
                                  XLNetConfig, XLNetForMultipleChoice_MT_general, XLNetTokenizer,
                                  RobertaConfig, RobertaForMultipleChoice_MT_general, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features, MAX_SEQ_LENGTHS,
                        output_modes, processors, GLUE_TASKS_NUM_LABELS)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (XLNetConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice_MT_general, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice_MT_general, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoice_MT_general, RobertaTokenizer),
}


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

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (total_n_tr_batches // args.gradient_accumulation_steps) + 1
    else:
        t_total = total_n_tr_batches // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=int(args.warmup_proportion * t_total)
                                     if args.warmup_proportion > 0 else args.warmup_steps,
                                     t_total=t_total)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      max_grad_norm=0,
    #                      t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = {}".format(','.join(map(str, args.per_gpu_train_batch_size))))
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
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3],
                      'task_id':        task_id}
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += inputs['input_ids'].size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.do_epoch_checkpoint:
            epoch_output_dir = os.path.join(args.output_dir, 'epoch_{}'.format(epoch))
            os.makedirs(epoch_output_dir, exist_ok=True)
            output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(epoch_output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        evaluate(args, model, tokenizer, prefix='', epoch=epoch, is_test=False)
        evaluate(args, model, tokenizer, prefix='', epoch=epoch, is_test=True)

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", epoch=0, is_test=False):
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
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3],
                          'task_id':        task_id}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if logits_all is None:
                logits_all = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                logits_all = np.append(logits_all, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if output_modes[eval_task] in ["classification", "multi-choice"]:
            preds = np.argmax(logits_all, axis=1)
        elif output_modes[eval_task] == "regression":
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
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
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
        features = convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTHS[task], tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
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
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_not_summary", action='store_true',
                        help="Set this flag if you do not want the summary step.")

    parser.add_argument("--per_gpu_train_batch_size", default='', type=str,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training. this will override warmup_steps")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_epoch_checkpoint",
                        default=False,
                        action='store_true',
                        help="Save checkpoint at every epoch")
    parser.add_argument('--same_linear_layer', type=int, default=0,
                        help="Whether to use same linear layer for multi-task")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        print("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower().split(',')
    args.per_gpu_train_batch_size = list(map(int, args.per_gpu_train_batch_size.split(',')))
    for task_name in args.task_name:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))
    args.data_dir = {task_name: data_dir_ for task_name, data_dir_ in zip(args.task_name, args.data_dir.split(','))}
    num_labels = [GLUE_TASKS_NUM_LABELS[task_name] for task_name in args.task_name]
    task_output_config = [(output_modes[task_name], num_label) for task_name, num_label in
                          zip(args.task_name, num_labels)]

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    ## we may need to remove the summary part
    if args.do_not_summary:
        setattr(config, 'summary_use_proj', False)
        setattr(config, 'summary_activation', '')
    logger.info("Updated model config %s", config)

    if 'roberta' in args.model_type:
        tokenizer = tokenizer_class(vocab_file=os.path.join(args.model_name_or_path, 'vocab.json'),
                                    merges_file=os.path.join(args.model_name_or_path, 'merges.txt'))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            task_output_config=task_output_config,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                            do_not_summary=args.do_not_summary,
                                            same_linear_layer=args.same_linear_layer)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else
                                                    args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            task_output_config=task_output_config,
                                            from_tf=bool('.ckpt' in args.model_name_or_path))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_datasets = [load_and_cache_examples(args, task_name, tokenizer, set_type='train')
                          for task_name in args.task_name]
        global_step, tr_loss = train(args, train_datasets, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, num_labels=num_labels, config=config)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    if args.do_eval and not args.do_train:
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))

        model.eval()
        epoch = args.num_train_epochs
        evaluate(args, model, tokenizer, epoch=epoch, is_test=False)
        evaluate(args, model, tokenizer, epoch=epoch, is_test=True)

    return None


if __name__ == "__main__":
    main()