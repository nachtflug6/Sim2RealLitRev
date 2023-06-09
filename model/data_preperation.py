import torch as pt
import numpy as np
import pandas as pd
from transformers import BertTokenizer


class BertDataset(pt.utils.data.Dataset):

    def __init__(self, df_data, df_label):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        df_data = pd.DataFrame(columns=['title', 'description'], data=df_data[['title', 'description']].values)
        df_data = df_data[df_data['description'] == df_data['description']]
        df_data = df_data[df_data['title'] == df_data['title']]

        df_data = df_data.applymap(lambda x: x if len(x.split()) < 512 else None)
        df_data = df_data[df_data['description'] == df_data['description']]
        df_data = df_data[df_data['title'] == df_data['title']]

        self.df_data = df_data.applymap(
            lambda x: tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors="pt"))
    # def classes(self):
    #     return self.labels

    # def __len__(self):
    #     return len(self.labels)

    # def get_batch_labels(self, idx):
    #     # Fetch a batch of labels
    #     return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.df_data.iloc[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        #batch_y = self.get_batch_labels(idx)

        return batch_texts