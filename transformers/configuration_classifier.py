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
""" CNN model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

from .configuration_utils import PretrainedConfig
import numpy as np

logger = logging.getLogger(__name__)


class ClassifierConfig(PretrainedConfig):
    def __init__(self,
                 batch_size=50,
                 class_size=2,
                 dropout_prob=0.1,
                 cnn_train=True,
                 **kwargs):
        super(ClassifierConfig, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.class_size = class_size
        self.dropout_prob = dropout_prob
        self.cnn_train = cnn_train
