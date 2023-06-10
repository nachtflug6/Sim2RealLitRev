from torch import nn
from transformers import BertModel


class BertBlock(nn.Module):

    def __init__(self, output_dims=2, dropout=0.5):
        super(BertBlock, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768, output_dims)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        final_layer = self.linear(pooled_output)

        return final_layer
