import torch
from torch.utils.data import DataLoader
from dataset import LegalDataset
import config
import tqdm
from Model import MyModel
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import utils
train_dataloader = DataLoader(LegalDataset(train=True), batch_size=config.batch_size, shuffle=True, num_workers=2)
loss_function = nn.NLLLoss()

test_dataloader = DataLoader(LegalDataset(train=False), batch_size=config.batch_size, shuffle=False, num_workers=2)
learning_rate = 0.001


def to_cuda(param_list):
    return [x.cuda() for x in param_list]


model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(config.num_epoch):
    total_loss = 0.
    model.train()
    for batch_data in tqdm.tqdm(train_dataloader, desc='Epoch %3d' % (i + 1)):
        curr_fact, term, law, accu = to_cuda(batch_data)
        optimizer.zero_grad()
        # print('term', term)
        # print('law', law)
        # print('accu', accu)
        pred1, pred2, pred3 = model(curr_fact)
        accu_loss = loss_function(pred1, accu)
        law_loss = loss_function(pred2, law)
        term_loss = loss_function(pred3, term)
        loss = accu_loss + law_loss + term_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    truth1, truth2, truth3 = [], [], []
    pred1_list, pred2_list, pred3_list = [], [], []
    for valid_data in test_dataloader:
        model.eval()
        valid_fact, term, law, accu = to_cuda(valid_data)
        pred1, pred2, pred3 = model(valid_fact)
        pred1_list.append(torch.argmax(pred1, dim=-1).cpu().data.numpy())
        pred2_list.append(torch.argmax(pred2, dim=-1).cpu().data.numpy())
        pred3_list.append(torch.argmax(pred3, dim=-1).cpu().data.numpy())
        truth1.append(accu.cpu().data.numpy())
        truth2.append(law.cpu().data.numpy())
        truth3.append(term.cpu().data.numpy())
    print('task1_acc', accuracy_score(np.concatenate(truth1), np.concatenate(pred1_list)))
    print('task2_acc', accuracy_score(np.concatenate(truth2), np.concatenate(pred2_list)))
    print('task3_acc', accuracy_score(np.concatenate(truth3), np.concatenate(pred3_list)))
    print('task1_precision_score', f1_score(np.concatenate(truth1), np.concatenate(pred1_list), average='macro'))
    print('task2_precision_score', f1_score(np.concatenate(truth2), np.concatenate(pred2_list), average='macro'))
    print('task3_precision_score', f1_score(np.concatenate(truth3), np.concatenate(pred3_list), average='macro'))



