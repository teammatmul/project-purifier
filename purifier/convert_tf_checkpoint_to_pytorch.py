# coding=utf-8

# Copyright 2019 team Purifier
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


# tf_checkpoint 경로
tf_checkpoint_path = "./data/"

# bert_config_file
bert_config_file = "./data/bert_config.json"

# save pytorch-model
pytorch_dump_path = "./data/torch/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)

