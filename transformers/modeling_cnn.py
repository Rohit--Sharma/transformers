import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, config, **kwargs):
        super(CNN, self).__init__()

        self.model = config.model
        self.batch_size = config.batch_size
        self.max_sent_len = config.max_sent_len
        self.word_dim = config.word_dim
        self.vocab_size = config.vocab_size
        self.class_size = config.class_size
        self.filters = config.filters
        self.filter_num = config.filter_num
        self.dropout_prob = config.dropout_prob
        self.in_channel = 1

        assert (len(self.filters) == len(self.filter_num))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
        if self.model == "static" or self.model == "non-static" or self.model == "multichannel":
            self.wv_matrix = config.wv_matrix
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            if self.model == "static":
                self.embedding.weight.requires_grad = False
            elif self.model == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2

        for i in range(len(self.filters)):
            conv = nn.Conv1d(self.in_channel, self.filter_num[i], self.word_dim * self.filters[i], stride=self.word_dim)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.filter_num), self.class_size)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.word_dim * self.max_sent_len)
        if self.model == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.word_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.filters[i] + 1)
                .view(-1, self.filter_num[i])
            for i in range(len(self.filters))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # logging.info('CNN fc shape: %s' % str(x.shape))
        # x = self.fc(x)

        return x
