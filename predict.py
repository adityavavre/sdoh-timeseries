import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from copy import deepcopy
import random

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
        self.layers.append(nn.Sequential(nn.Linear(hidden_dims[-1], 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--type', type=str, default='no_sdoh', help='Whether to include SDoH annotation in training data or not')
    args = parser.parse_args()

    length_of_stay_df = pd.read_csv(f'/home/av38898/projects/sdoh/data/length_of_stay_features.csv')

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
        X = length_of_stay_df[input_feature_columns_sdoh]
    elif args.type == 'only_sdoh':
        X = length_of_stay_df[input_feature_columns_only_sdoh]
    elif args.type == 'no_sdoh':
        X = length_of_stay_df[input_feature_columns]
    else:
        print(f'Type not recognized!')
        exit()

    Y = length_of_stay_df[['length_of_stay']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state= 0)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.10, random_state=0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Dev set has {} samples.".format(X_dev.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    dataset = list(zip([torch.tensor(i) for i in X_train.values.tolist()], y_train.values[:, 0].tolist()))
    dev_dataset = list(zip([torch.tensor(i) for i in X_dev.values.tolist()], y_dev.values[:, 0].tolist()))
    test_dataset = list(zip([torch.tensor(i) for i in X_test.values.tolist()], y_test.values[:, 0].tolist()))

    model = FFN(hidden_dims=[len(X_train.columns), 16], dropout=0.3)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    LR = 5e-4
    NUM_EPOCHS = 400
    BATCH_SIZE = 128

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss().to(device)

    best_dev_R2 = -np.inf
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
            label = torch.tensor([b[1] for b in batch], dtype=torch.float).to(device)
            pred = model(inputs)
            loss = loss_fn(pred.squeeze(), label)

            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().cpu().item())
            preds.extend(pred.detach().cpu().squeeze().tolist())
            labels.extend(label.detach().cpu().tolist())

        print(f'Loss: {np.mean(losses)} R2: {r2_score(labels, preds)}')

        model.eval()
        dev_preds = []
        dev_labels = []
        with torch.no_grad():
            for i in range(0, len(dev_dataset), BATCH_SIZE):
                batch = dev_dataset[i:i+BATCH_SIZE]
                inputs = torch.stack([b[0] for b in batch]).type(torch.float).to(device)
                label = torch.tensor([b[1] for b in batch], dtype=torch.float).to(device)
                pred = model(inputs)
                dev_preds.extend(pred.detach().cpu().squeeze().tolist())
                dev_labels.extend(label.detach().cpu().tolist())
            
            r2 = r2_score(dev_labels, dev_preds)
            print(f'Dev R2: {r2}')
            if r2 > best_dev_R2:
                best_dev_R2 = r2
                best_model = deepcopy(model)

    print(f'Best Dev R2: {best_dev_R2}')

    # Best Test R2: 0.28432222941207685  SDoH R2: 0.2854337584167348


best_model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for i in range(0, len(test_dataset), BATCH_SIZE):
        batch = test_dataset[i:i+BATCH_SIZE]
        inputs = torch.stack([b[0] for b in batch]).type(torch.float).to(device)
        label = torch.tensor([b[1] for b in batch], dtype=torch.float).to(device)
        pred = best_model(inputs)
        test_preds.extend(pred.detach().cpu().squeeze().tolist())
        test_labels.extend(label.detach().cpu().tolist())
r2 = r2_score(test_labels, test_preds)
print(f'Test R2: {r2}')

# only_sdoh 
# Best Dev R2: -0.00031785727373523365
# Test R2: -0.00019748215520087875

# sdoh
# Best Dev R2: 0.26228114729675167
# Test R2: 0.274068163217713

# no_sdoh
# Best Dev R2: 0.25710850092803306
# Test R2: 0.272258979209132
            