import json
import torch
from torch.utils.data import Dataset
class TripletDataset(Dataset):
    def __init__(self, anchor_df, positive_df, negative_df):
        self.anchor   = anchor_df
        self.positive = positive_df
        self.negative = negative_df

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        a_vec = torch.tensor(json.loads(self.anchor.iloc[idx]),   dtype=torch.float32)
        p_vec = torch.tensor(json.loads(self.positive.iloc[idx]), dtype=torch.float32)
        n_vec = torch.tensor(json.loads(self.negative.sample(1).iloc[0]), dtype=torch.float32)
        return a_vec, p_vec, n_vec