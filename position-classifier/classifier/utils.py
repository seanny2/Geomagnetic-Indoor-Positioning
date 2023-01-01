import torch
import pandas as pd

# 샘플 데이터 -> 텐서
def load_data(pattern):
    data_x = pd.read_table("./data/train/inputs.txt", sep=",")
    data_y = pd.read_table("./data/train/outputs.txt")
    x = torch.from_numpy(data_x[['X', 'Y', 'Z', 'F']].values).float()
    y = torch.from_numpy(data_y['class'].values)

    x = x.view(int(x.size(0)/pattern), pattern, -1)
    
    return x, y

# train / valid / test data split
def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.shape[0] * train_ratio)
    valid_cnt = x.shape[0] - train_cnt
    
    indices = torch.randperm(x.shape[0])
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y