import time
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from model import NARM
from DataLoader import collate_fn, RecSysDataset, load_data
from metrics import evaluate
from utils import device, topk, here
from torch.optim.lr_scheduler import StepLR


def train(data_path, epoch, batch_size, hidden_size, embedding_dim, testing=False):
    print('Loading data...')
    train, valid, test = load_data(data_path, valid_portion=0.1)
    mrr_list, recall_list, loss_list = [], [], []

    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    n_items = 37484

    model = NARM(n_items, hidden_size=hidden_size, embedding_dim=embedding_dim, batch_size=batch_size).to(device)

    if testing:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        recall, mrr = validate(test_loader, model)
        print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(topk, recall, topk, mrr))
        return

    optimizer = optim.Adam(model.parameters(), 1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)

    for e in tqdm(range(epoch)):
        # train for one epoch
        scheduler.step(epoch=e)
        sum_loss = trainForEpoch(train_loader, model, optimizer, e, epoch, criterion)

        print('[TRAIN] epoch %d/%d avg loss %.4f'
              % (epoch + 1, epoch, sum_loss / len(train_loader.dataset)))

        recall, mrr = validate(valid_loader, model)
        recall_list.append(recall)
        mrr_list.append(mrr)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(e, topk, recall, topk,
                                                                                 mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, here + f'/checkpoint_{e}.pth.tar')
    return mrr_list, recall_list, loss_list


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion):
    model.train()

    sum_epoch_loss = 0

    for i, (seq, target, lens) in enumerate(train_loader):
        seq = seq.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

    return sum_epoch_loss


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim=1)
            recall, mrr = evaluate(logits, target, k=topk)
            recalls.append(recall)
            mrrs.append(mrr)

    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr


if __name__ == "__main__":
    params = {
        'epoch': 100,
        'batch_size': 512,
        'hidden_size': 100,
        'embedding_dim': 50}
    print('-' * 50)
    print("Parameters :")
    print(params)
    print('-' * 50)
    mrr_list, recall_list, loss_list = train(data_path='yoochoose1_64/', **params)
