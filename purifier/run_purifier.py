# coding=utf-8

# Copyright 2019 team Purifier
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import math
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import matthews_corrcoef, f1_score

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling_purifier import BertForSequenceClassification, BertConfig
from mask_tokenizer import BertTokenizer, BasicTokenizer
from optimization import BertAdam, WarmupLinearSchedule

"""====== set parameter (변수설정) =======
gradient_accumulation_steps = 1
train_batch_size = 32
seed = 42
local_rank = -1
no_cuda = True
fp16 = False
do_train = False
do_eval = True
output_dir = './data/ouput/'
vocab_file = './data/vocab_korea.txt'
task_name = 'Puri'
do_lower_case = True
data_dir = './data/'
max_seq_length = 128
# num_train_epochs = 0.1
# warmup_proportion = 0.1
# learning_rate = 5e-5
# ====================================="""

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
        
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    puri_ids_list = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, puri_ids = tokenizer.tokenize(example.text_a)
        puri_ids_list.append(puri_ids)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # also we mask SEP token since we don't need to classify next sentence
        input_mask = [1] * (len(input_ids)-1)
        input_mask += [0]

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features, puri_ids_list

def convert_single_example_to_feature(example, max_seq_length, tokenizer, output_mode):
    '''Convert single InputExample to InputFeature for predict'''
    # we use puri_ids_list for masking toxic expression
    tokens_a, puri_ids_list = tokenizer.tokenize(example.text_a)
    tokens_b = None
    
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    
    # convert tokens to vocab index
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    # also we mask SEP token since we don't need to classify next sentence
    input_mask = [1] * (len(input_ids)-1)
    input_mask += [0]
    
    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    
    # check each element's length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    # we only want to predict task
    label_id = None
    
    # convert example to feature
    feature = InputFeatures(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           label_id=label_id)
    
    return feature, puri_ids_list

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

class PuriProcessor(DataProcessor):
    """Processor for the Puri data set (GLUE version)."""

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
        """Create examples train / dev"""       
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = None
            label = line[0]
            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            
        return examples
    
    def create_single_example(self, text):        
        """Creates single exmaple for predicting a single sentence"""
        guid = 0
        text_a = text
        text_b = None
        label = None
        example = InputExample(guid=guid, 
                               text_a = text_a, 
                               text_b=text_b, 
                               label=None)

        return example

    def create_list_example(self, text_list):
        """Creates examples for list object"""
        examples = []
        set_type = "ltest"
        for (i, text) in enumerate(text_list):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            text_b = None
            label = '0'
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        
        return examples

def single_sentence_masking_percent(text, model):
    """Run text predict model and toxic text masking by loop.

    Inputs:
        `text`: user string input
        `model`: fine-tunned model. This case we use purifier model

    Model params:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `front_pooler` : choose to apply tanh activation function on encoder process . Default: `True`.
        `query`, `key`, `value` : select layers for input of puri attention. selected_layers is indices of index which 0 index means embedding_output.
        `query_att` :True or False
        `key_att` : True or False
        `multi_head` : choose to apply multi-head attention. Default: `True`.
        `dropout` : choose to apply dropout to attention_probs. Default: `False`.
        `back_pooler` : choose to apply tanh activation function after puri layer . Default: `True`.

    Outputs:
        `final_result[0]` : predict result of model
                          0 means not toxic text
                          1 means toxic text
        if `final_result[0]` is 0:
            `text` : no maksing sentence
        if `final_result[0]` is 1:
            `text` : maksing sentence
    ```
    """
    
    # set parameter
    # 변수설정
    seed = 42
    local_rank = -1
    no_cuda = True
    fp16 = False
    do_train = False
    do_eval = True
    output_dir = './data/ouput/'
    vocab_file = './data/vocab_korea.txt'
    task_name = 'Puri'
    do_lower_case = True
    data_dir = './data/'
    max_seq_length = 128

    # set parameter related with processor
    # processor 관련
    processors = {"puri": PuriProcessor}
    output_modes = {"puri": "classification"}
    task_name = task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()

    # set device option
    # device 설정
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    # declare tokenizer which is in mask_tokenizer
    # mask 토크나이저 선언
    tokenizer = BertTokenizer(vocab_file)

    model.to(device)
    model.eval()

    result = 1
    for_result =1

    # loop for masking token when our model predicts that the text has toxic multiple expression
    # 욕설 마스킹을 위한 반복
    while result:
        example = processor.create_single_example(text)
        feature, puri_ids_list = convert_single_example_to_feature(example, 128, tokenizer, output_mode)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
        input_mask  = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids  = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            logits, cls_info = model(input_ids, segment_ids, input_mask, labels=None,
                                     front_pooler=True,
                                     query=[12],
                                     key=[1,2,3],
                                     value=[1,2,3],
                                     query_att=True,
                                     key_att=True,
                                     multi_head=False,
                                     dropout=False,
                                     back_pooler=True)

        preds = []

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        preds = preds[0]
        result = np.argmax(preds, axis=1)

        if for_result:
            final_result = result
            for_result=0

        # if text is not toxic we escape loop
        if result == 0:
            return text, final_result[0]
        
        basic_tokenized = BasicTokenizer(do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")).tokenize(text)
        
        # mask toxic expression which has most attention prob value
        toxic_ids = list(cls_info['probs'][0][0]).index(max(cls_info['probs'][0][0]))
        basic_tokenized[puri_ids_list[toxic_ids-1]] = "*"
        
        # if text and sentence after masking process is equal we escape loop
        if text == " ".join(basic_tokenized):
            return text, final_result[0]
        
        text = " ".join(basic_tokenized)
        