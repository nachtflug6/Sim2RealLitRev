import torch as pt
import numpy as np
import pandas as pd
from transformers import BertTokenizer


class ScopusDataset(pt.utils.data.Dataset):

    def __init__(self, df_data):
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

    def __len__(self):
        return len(self.df_data.index)

    # def get_batch_labels(self, idx):
    #     # Fetch a batch of labels
    #     return np.array(self.labels[idx])

    def get_row(self, idx):
        # Fetch a batch of inputs
        return self.df_data.iloc[idx]

    def __getitem__(self, idx):
        row = self.get_row(idx)

        title = row['title']
        abstract = row['description']

        out_tuple = {'title': (title['input_ids'][0], title['attention_mask'][0]),
                     'abstract': (abstract['input_ids'][0], abstract['attention_mask'][0])}


        return out_tuple
