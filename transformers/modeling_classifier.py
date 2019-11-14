import torch
import logging
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LinearClassifier(nn.Module):
    def __init__(self, config, **kwargs):
        super(LinearClassifier, self).__init__()

        self.batch_size = config.batch_size
        self.class_size = config.class_size
        self.dropout_prob = config.dropout_prob
        self.cnn_train = config.cnn_train

        self.fc_size = 768 + 300 if self.cnn_train else 768

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.fc_size, self.class_size)

        self.criterion = CrossEntropyLoss()

    def forward(self, bert_last_layer, cnn_last_layer, labels):
        bert_last_layer = self.dropout(bert_last_layer)

        if self.cnn_train:
            concat_emb = torch.cat([bert_last_layer, cnn_last_layer], 1)
        else:
            concat_emb = bert_last_layer

        logits = self.classifier(concat_emb)

        loss = self.criterion(logits.view(-1, self.class_size), labels.view(-1))
        return loss, logits
