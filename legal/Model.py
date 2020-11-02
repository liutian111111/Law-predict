import torch
import torch.nn as nn
import config
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.word_emb = nn.Embedding(config.word_num, config.word_hidden)
        self.feature_num = 75
        self.conv1 = nn.Conv2d(1, self.feature_num, (2, config.word_hidden))
        self.conv2 = nn.Conv2d(1, self.feature_num, (3, config.word_hidden))
        self.conv3 = nn.Conv2d(1, self.feature_num, (3, config.word_hidden))
        self.conv4 = nn.Conv2d(1, self.feature_num, (3, config.word_hidden))
        self.pool1 = nn.MaxPool2d((config.seq_len - 1, 1))
        self.pool2 = nn.MaxPool2d((config.seq_len - 2, 1))
        self.pool3 = nn.MaxPool2d((config.seq_len - 2, 1))
        self.pool4 = nn.MaxPool2d((config.seq_len - 2, 1))

        self.l1 = nn.Linear(self.feature_num * 4, config.hid_size)
        self.l2 = nn.Linear(self.feature_num * 4, config.hid_size)
        self.l3 = nn.Linear(self.feature_num * 4, config.hid_size)
        self.task1_linear = nn.Linear(config.hid_size, config.num_accu)
        self.task2_linear = nn.Linear(config.hid_size, config.num_law)
        self.task3_linear = nn.Linear(config.hid_size, config.num_term)
        nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.word_emb(x).unsqueeze(1)  # B * S * C
        x1 = self.pool1(F.elu(self.conv1(x))).view(-1, self.feature_num)
        x2 = self.pool2(F.elu(self.conv2(x))).view(-1, self.feature_num)
        x3 = self.pool3(F.elu(self.conv3(x))).view(-1, self.feature_num)
        x4 = self.pool4(F.elu(self.conv4(x))).view(-1, self.feature_num)
        feature = torch.cat([x1, x2, x3, x4], dim=-1)
        feature = F.dropout(feature, p=0.5, training=self.training)
        task1_pred = self.task1_linear(F.elu(self.l1(feature)))
        task2_pred = self.task2_linear(F.elu(self.l2(feature)))
        task3_pred = self.task3_linear(F.dropout(F.elu(self.l3(feature)), p=0.5, training=self.training))
        return F.log_softmax(task1_pred, dim=1), F.log_softmax(task2_pred, dim=1), F.log_softmax(task3_pred, dim=1)


