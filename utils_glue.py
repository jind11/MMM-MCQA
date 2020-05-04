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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import json
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
    def _read_tsv(cls, input_file, quotechar=None, remove_header=False):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            if remove_header:
                next(reader)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DreamProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/{}.json".format(set_type), 'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    text_a = '\n'.join(data[i][0])
                    text_c = data[i][1][j]["question"]
                    options = []
                    for k in range(len(data[i][1][j]["choice"])):
                        options.append(data[i][1][j]["choice"][k])
                    answer = data[i][1][j]["answer"]
                    label = str(options.index(answer))
                    for k in range(len(options)):
                        guid = "%s-%s-%s" % (set_type, i, k)
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=options[k], label=label, text_c=text_c))

        return examples


class UniProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "test")

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/{}.json".format(set_type), 'r') as f:
            data = json.load(f)
            for guid, example in data.items():
                context = example['context']
                question = example['question']
                options = example['options']
                label = example['answer_idx']
                for k in range(len(options)):
                    guid = "%s-%s" % (set_type, guid)
                    examples.append(
                        InputExample(guid=guid, text_a=context, text_b=options[k], label=label, text_c=question))

        return examples


class UniIndProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "test")

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/{}.json".format(set_type), 'r') as f:
            data = json.load(f)
            for guid, example in data.items():
                context = example['context']
                question = example['question']
                label = example['answer_idx']
                guid = "%s-%s" % (set_type, guid)
                examples.append(
                    InputExample(guid=guid, text_a=context, text_b=example['answer'], label=label, text_c=question))

        return examples


class RaceProcessor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'RACE'

    def _read_samples(self, data_dir, set_type, level=None):
        # if self.level == None:
        #     data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
        #                  '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        # else:
        # data_dirs = ['{}/{}/{}'.format(data_dir, set_type, self.level)]
        if level is None:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
                         '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        else:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, level)]

        examples = []
        example_id = 0
        for data_dir in data_dirs:
            # filenames = glob.glob(data_dir + "/*txt")
            filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    article = data_raw['article']
                    for i in range(len(data_raw['answers'])):
                        example_id += 1
                        truth = str(ord(data_raw['answers'][i]) - ord('A'))
                        question = data_raw['questions'][i]
                        options = data_raw['options'][i]
                        for k in range(len(options)):
                            guid = "%s-%s-%s" % (set_type, example_id, k)
                            option = options[k]
                            examples.append(
                                    InputExample(guid=guid, text_a=article, text_b=option, label=truth,
                                                 text_c=question))

        return examples


class ToeflProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'TOEFL'

    def _read_samples(self, data_dir, set_type):

        examples = []
        example_id = 0
        data_dir = join(data_dir, set_type)
        filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        for filename in filenames:
            example_id += 1
            with open(filename, 'r', encoding='utf-8') as fpr:
                article = []
                options = []
                for line in fpr:
                    line_tag, _, text = line.strip().partition(' ')
                    if line_tag == 'SENTENCE':
                        article.append(text)
                    elif line_tag == 'QUESTION':
                        question = text
                    elif line_tag == 'OPTION':
                        options.append(text[:-2])
                        if text[-1] == '1':
                            truth = text[:-2]
                    else:
                        raise KeyError('no such tag!')
                article = '\n'.join(article)
                truth = str(options.index(truth))
                for k, option in enumerate(options):
                    guid = "%s-%s-%s" % (set_type, example_id, k)
                    examples.append(
                            InputExample(guid=guid, text_a=article, text_b=option, label=truth,
                                         text_c=question))

        return examples


class MCTest160Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest160'

    def _read_samples(self, data_dir, set_type):

        with open(join(data_dir, "mc160.{}.tsv".format(set_type)), 'r', encoding='utf-8') as fpr:
            articles = []
            questions = [[]]
            options = [[]]
            for line in fpr:
                line = line.strip().split('\t')
                assert len(line) == 23
                articles.append(line[2].replace("\\newline", " "))
                for idx in range(3, 23, 5):
                    questions[-1].append(line[idx].partition(":")[-1][1:])
                    options[-1].append(line[idx+1:idx+5])
                questions.append([])
                options.append([])

        with open(join(data_dir, "mc160.{}.ans".format(set_type)), 'r', encoding='utf-8') as fpr:
            answers = []
            for line in fpr:
                line = line.strip().split('\t')
                answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        for article, question, option, answer in zip(articles, questions, options, answers):
            for ques, opt, ans in zip(question, option, answer):
                example_id += 1
                for k, op in enumerate(opt):
                    guid = "%s-%s-%s" % (set_type, example_id, k)
                    examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))

        return examples


class MCTest500Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest500'

    def _read_samples(self, data_dir, set_type):

        with open(join(data_dir, "mc500.{}.tsv".format(set_type)), 'r', encoding='utf-8') as fpr:
            articles = []
            questions = [[]]
            options = [[]]
            for line in fpr:
                line = line.strip().split('\t')
                assert len(line) == 23
                articles.append(line[2].replace("\\newline", " "))
                for idx in range(3, 23, 5):
                    questions[-1].append(line[idx].partition(":")[-1][1:])
                    options[-1].append(line[idx+1:idx+5])
                questions.append([])
                options.append([])

        with open(join(data_dir, "mc500.{}.ans".format(set_type)), 'r', encoding='utf-8') as fpr:
            answers = []
            for line in fpr:
                line = line.strip().split('\t')
                answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        for article, question, option, answer in zip(articles, questions, options, answers):
            for ques, opt, ans in zip(question, option, answer):
                example_id += 1
                for k, op in enumerate(opt):
                    guid = "%s-%s-%s" % (set_type, example_id, k)
                    examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))

        return examples


class MCTestProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest'

    def _read_samples(self, data_dir, set_type):

        articles = []
        questions = [[]]
        options = [[]]
        for filename in [join(data_dir, "mc160.{}.tsv".format(set_type)),
                         join(data_dir, "mc500.{}.tsv".format(set_type))]:
            with open(filename, 'r', encoding='utf-8') as fpr:
                for line in fpr:
                    line = line.strip().split('\t')
                    assert len(line) == 23
                    articles.append(line[2].replace("\\newline", " "))
                    for idx in range(3, 23, 5):
                        questions[-1].append(line[idx].partition(":")[-1][1:])
                        options[-1].append(line[idx + 1:idx + 5])
                    questions.append([])
                    options.append([])

        answers = []
        for filename in [join(data_dir, "mc160.{}.ans".format(set_type)),
                         join(data_dir, "mc500.{}.ans".format(set_type))]:
            with open(filename, 'r', encoding='utf-8') as fpr:
                for line in fpr:
                    line = line.strip().split('\t')
                    answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        for article, question, option, answer in zip(articles, questions, options, answers):
            for ques, opt, ans in zip(question, option, answer):
                example_id += 1
                for k, op in enumerate(opt):
                    guid = "%s-%s-%s" % (set_type, example_id, k)
                    examples.append(
                        InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                     text_c=ques))

        return examples


class MCScriptProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_dataset_name(self):
        return 'MCScript'

    def _read_samples(self, data_dir, set_type):

        tree = ET.parse(join(data_dir, '{}-data.xml'.format(set_type)))
        root = tree.getroot()
        example_id = 0
        examples = []
        for ele in root.iter('instance'):
            for subele in ele:
                if subele.tag == 'text':
                    article = subele.text
                else:
                    for subsubele in subele.iter('question'):
                        example_id += 1
                        question = subsubele.attrib['text']
                        options = []
                        for answer in subsubele:
                            options.append(answer.attrib['text'])
                            if answer.attrib['correct'] == 'True':
                                label = answer.attrib['id']
                        for k, option in enumerate(options):
                            guid = "%s-%s-%s" % (set_type, example_id, k)
                            examples.append(
                                InputExample(guid=guid, text_a=article, text_b=option, label=label,
                                             text_c=question))

        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# class MnliProcessor(DataProcessor):
#     """Processor for the MultiNLI data set."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, 'multinli_1.0', "multinli_1.0_train.txt")) +
#             self._read_tsv(os.path.join(data_dir, 'multinli_1.0', "multinli_1.0_dev_matched.txt"),
#                            remove_header=True), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, 'multinli_1.0', "multinli_1.0_dev_matched.txt")),
#             "dev_matched")
#
#     def get_labels(self):
#         """See base class."""
#         return ["contradiction", "entailment", "neutral"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         parentheses_table = str.maketrans({'(': None, ')': None})
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[0].lower()
#             if label == '-':
#                 continue
#             # Remove '(' and ')' from the premises and hypotheses.
#             text_a = text_a.translate(parentheses_table)
#             text_b = text_b.translate(parentheses_table)
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"))
            # self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"), remove_header=True), "train")
            , "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")) +
            self._read_tsv(os.path.join(data_dir, "dev.tsv"), remove_header=True), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class YelpProcessor(DataProcessor):
    """Processor for the Yelp data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "sentiment.train.0"), "train") + \
               self._create_examples(os.path.join(data_dir, "sentiment.train.1"), "train")


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "sentiment.dev.0"), "dev") + \
               self._create_examples(os.path.join(data_dir, "sentiment.dev.1"), "dev")

    def test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "sentiment.test.0"), "test") + \
               self._create_examples(os.path.join(data_dir, "sentiment.test.1"), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, data_path, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label = "1" if data_path.endswith('1') else "0"
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for (i, line) in enumerate(f):
                guid = "%s-%s" % (set_type, i)
                text_a = line.strip()
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# def convert_examples_to_features(examples, label_list, max_seq_length,
#                                  tokenizer, output_mode,
#                                  cls_token_at_end=False, pad_on_left=False,
#                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
#                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
#                                  cls_token_segment_id=1, pad_token_segment_id=0,
#                                  mask_padding_with_zero=True, do_lower_case=False,
#                                  is_multi_choice=True):
#     """ Loads a data file into a list of `InputBatch`s
#         `cls_token_at_end` define the location of the CLS token:
#             - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
#             - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
#         `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
#     """
#
#     label_map = {label : i for i, label in enumerate(label_list)}
#
#     if is_multi_choice:
#         features = [[]]
#     else:
#         features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens_a = tokenizer.tokenize(example.text_a)
#
#         tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(tokens_a) > max_seq_length - 2:
#                 tokens_a = tokens_a[:(max_seq_length - 2)]
#
#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids:   0   0   0   0  0     0   0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambiguously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.
#         tokens = tokens_a + [sep_token]
#         segment_ids = [sequence_a_segment_id] * len(tokens)
#
#         if tokens_b:
#             tokens += tokens_b + [sep_token]
#             segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
#
#         if cls_token_at_end:
#             tokens = tokens + [cls_token]
#             segment_ids = segment_ids + [cls_token_segment_id]
#         else:
#             tokens = [cls_token] + tokens
#             segment_ids = [cls_token_segment_id] + segment_ids
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         padding_length = max_seq_length - len(input_ids)
#         if pad_on_left:
#             input_ids = ([pad_token] * padding_length) + input_ids
#             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
#             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
#         else:
#             input_ids = input_ids + ([pad_token] * padding_length)
#             input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#             segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         if output_mode == "classification":
#             label_id = label_map[example.label]
#         elif output_mode == "regression":
#             label_id = float(example.label)
#         else:
#             raise KeyError(output_mode)
#
#         # if ex_index < 5:
#         #     logger.info("*** Example ***")
#         #     logger.info("guid: %s" % (example.guid))
#         #     logger.info("tokens: %s" % " ".join(
#         #             [str(x) for x in tokens]))
#         #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#         #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#         #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#         #     logger.info("label: %s (id = %d)" % (example.label, label_id))
#
#         features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_mask=input_mask,
#                               segment_ids=segment_ids,
#                               label_id=label_id))
#     return features


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 do_lower_case=False, is_multi_choice=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    if is_multi_choice:
        features = [[]]
    else:
        features = []
    for (ex_index, example) in enumerate(examples):
        if do_lower_case:
            example.text_a = example.text_a.lower()
            example.text_b = example.text_b.lower()
            example.text_c = example.text_c.lower()

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        elif example.text_b and not example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_c:
            tokens_b += [sep_token] + tokens_c

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

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
            if len(features[-1]) == num_labels:
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

    return features


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'snli':
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'race':
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'dream':
        return {"acc": simple_accuracy(preds, labels)}
    else:
        return {"acc": simple_accuracy(preds, labels)}

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "snli": SnliProcessor,
    "dream": DreamProcessor,
    "race": RaceProcessor,
    "toefl": ToeflProcessor,
    "mctest": MCTestProcessor,
    "mctest160": MCTest160Processor,
    "mctest500": MCTest500Processor,
    "mcscript": MCScriptProcessor,
    "yelp": YelpProcessor,
    "uni": UniProcessor,
    "uniind": UniIndProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "snli": "classification",
    "dream": "multi-choice",
    "race": 'multi-choice',
    "toefl": 'multi-choice',
    "mctest": 'multi-choice',
    "mctest160": 'multi-choice',
    "mctest500": 'multi-choice',
    "mcscript": 'multi-choice',
    "yelp": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "dream": 3,
    "race": 4,
    "mnli-mm": 3,
    "toefl": 4,
    "mctest": 4,
    "mctest160": 4,
    "mctest500": 4,
    "mcscript": 2,
    "yelp": 2,
}

MAX_SEQ_LENGTHS = {
    "race": 512,
    "dream": 512,
    "mnli": 128,
    "snli": 128,
    "cola": 128,
    "mrpc": 128,
    "sst-2": 128,
    "sts-b": 128,
    "qqp": 128,
    "rte": 128,
    "wnli": 128,
    "qnli": 128,
    "mnli-mm": 128,
    "toefl": 512,
    "mctest": 512,
    "mctest160": 512,
    "mctest500": 512,
    "mcscript": 512,
    "yelp": 128,
}