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
from gensim.models.keyedvectors import KeyedVectors

logger = logging.getLogger(__name__)


class CNNConfig(PretrainedConfig):
    def __init__(self,
                 model='non-static',
                 batch_size=50,
                 max_sent_len=500,
                 word_dim=300,
                 vocab_size=30522,
                 class_size=2,
                 filters=[3, 4, 5],
                 filter_num=[100, 100, 100],
                 dropout_prob=0.5,
                 word_emb_path='',
                 **kwargs):
        super(CNNConfig, self).__init__(**kwargs)
        self.model = model
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.filters = filters
        self.filter_num = filter_num
        self.dropout_prob = dropout_prob
        self.word_emb_path = word_emb_path

        # Load word2vec
        word_vectors = KeyedVectors.load_word2vec_format(self.word_emb_path, binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        self.wv_matrix = wv_matrix
