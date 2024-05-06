import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from copy import deepcopy
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import optim

import argparse

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

class FFN(nn.Module):
    def __init__(self, hidden_dims: List[int], dropout: float = 0):
        super(FFN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.layers.append(nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), self.activation, self.dropout))
        for i in range(1, len(hidden_dims)-1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), self.activation, self.dropout))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dims[-1], 2)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--type', type=str, default='no_sdoh', help='Whether to include SDoH annotation in training data or not')
    args = parser.parse_args()

    with open('data/LOS_label/X_train_LOS_label.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/LOS_label/y_train_LOS_label.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/LOS_label/X_test_LOS_label.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/LOS_label/y_test_LOS_label.pkl', 'rb') as f:
        y_test = pickle.load(f)

    input_feature_columns = ['blood', 'circulatory', 'congenital',
        'digestive', 'endocrine', 'genitourinary', 'infectious', 'injury',
        'mental', 'misc', 'muscular', 'neoplasms', 'nervous', 'pregnancy',
        'prenatal', 'respiratory', 'skin', 'GENDER', 'ICU', 'NICU',
        'ADM_ELECTIVE', 'ADM_EMERGENCY', 'ADM_URGENT', 'INS_Government',
        'INS_Medicaid', 'INS_Medicare', 'INS_Private', 'INS_Self Pay',
        'REL_NOT SPECIFIED', 'REL_RELIGIOUS', 'REL_UNOBTAINABLE', 'ETH_ASIAN',
        'ETH_BLACK/AFRICAN AMERICAN', 'ETH_HISPANIC/LATINO',
        'ETH_OTHER/UNKNOWN', 'ETH_WHITE', 'AGE_middle_adult', 'AGE_newborn',
        'AGE_senior', 'AGE_young_adult', 'MAR_DIVORCED', 'MAR_LIFE PARTNER',
        'MAR_MARRIED', 'MAR_SEPARATED', 'MAR_SINGLE', 'MAR_UNKNOWN (DEFAULT)',
        'MAR_WIDOWED']

    input_feature_columns_only_sdoh = ['sdoh_community-present', 'sdoh_economics', 'sdoh_tobacco']
    input_feature_columns_sdoh = input_feature_columns+input_feature_columns_only_sdoh


    if args.type == 'sdoh':
        X_train = X_train[input_feature_columns_sdoh]
        X_test = X_test[input_feature_columns_sdoh]
    elif args.type == 'only_sdoh':
        X_train = X_train[input_feature_columns_only_sdoh]
        X_test = X_test[input_feature_columns_only_sdoh]
    elif args.type == 'no_sdoh':
        X_train = X_train[input_feature_columns]
        X_test = X_test[input_feature_columns]
    else:
        print(f'Type not recognized!')
        exit()

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.10, random_state=0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Dev set has {} samples.".format(X_dev.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    dataset = list(zip([torch.tensor(i) for i in X_train.values.tolist()], y_train.values[:, 0].tolist()))
    dev_dataset = list(zip([torch.tensor(i) for i in X_dev.values.tolist()], y_dev.values[:, 0].tolist()))
    test_dataset = list(zip([torch.tensor(i) for i in X_test.values.tolist()], y_test.values[:, 0].tolist()))

    model = FFN(hidden_dims=[len(X_train.columns), 32, 8], dropout=0.2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    LR = 1e-4
    NUM_EPOCHS = 400
    BATCH_SIZE = 512

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_dev_loss = np.inf
    best_model = model

    for e in range(NUM_EPOCHS):
        model.train()
        print(f'Epoch: {e+1}/{NUM_EPOCHS}')
        random.shuffle(dataset)
        losses = []
        preds = []
        labels = []
        for i in range(0, len(dataset), BATCH_SIZE):
            optimizer.zero_grad()

            batch = dataset[i:i+BATCH_SIZE]
            inputs = torch.stack([b[0] for b in batch]).type(torch.float).to(device)
            label = torch.tensor([b[1] for b in batch]).type(torch.LongTensor).to(device)
            pred = model(inputs)
            loss = loss_fn(pred.squeeze(), label)

            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().cpu().item())
            preds.extend(torch.argmax(pred, dim=-1).detach().cpu().squeeze().tolist())
            labels.extend(label.detach().cpu().tolist())

        print(f'Train Loss: {np.mean(losses)} Train Accuracy: {accuracy_score(labels, preds)}')

        model.eval()
        dev_preds = []
        dev_labels = []
        dev_losses = []
        with torch.no_grad():
            for i in range(0, len(dev_dataset), BATCH_SIZE):
                batch = dev_dataset[i:i+BATCH_SIZE]
                inputs = torch.stack([b[0] for b in batch]).type(torch.float).to(device)
                label = torch.tensor([b[1] for b in batch]).type(torch.LongTensor).to(device)
                pred = model(inputs)
                loss = loss_fn(pred.squeeze(), label)

                dev_losses.append(loss.detach().cpu().item())
                dev_preds.extend(torch.argmax(pred, dim=-1).detach().cpu().squeeze().tolist())
                dev_labels.extend(label.detach().cpu().tolist())
            
            dev_loss = np.mean(dev_losses)
            print(f'Dev Loss: {dev_loss} Dev Accuracy: {accuracy_score(dev_labels, dev_preds)}')

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_model = deepcopy(model)


best_model.eval()
test_preds = []
test_labels = []
test_preds_proba = []
with torch.no_grad():
    for i in range(0, len(test_dataset), BATCH_SIZE):
        batch = test_dataset[i:i+BATCH_SIZE]
        inputs = torch.stack([b[0] for b in batch]).type(torch.float).to(device)
        label = torch.tensor([b[1] for b in batch]).type(torch.LongTensor).to(device)
        pred = best_model(inputs)

        test_preds_proba.extend(pred[:, 1].detach().cpu().squeeze().tolist())
        test_preds.extend(torch.argmax(pred, dim=-1).detach().cpu().squeeze().tolist())
        test_labels.extend(label.detach().cpu().tolist())

test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
test_auc = roc_auc_score(test_labels, test_preds_proba)
print(f'Test Acc: {test_acc} \nTest F1: {test_f1}  \nTest AUC: {test_auc}')

# no_sdoh
# Test Acc: 0.6730304187348861 
# Test F1: 0.6675294422156076  
# Test AUC: 0.7441109010602607

# sdoh
# Test Acc: 0.6764668448517246 
# Test F1: 0.6625066383430696  
# Test AUC: 0.7439540217163519

# only_sdoh 
# Test Acc: 0.5037546137202494 
# Test F1: 0.5087564570996598  
# Test AUC: 0.4961380530450469
