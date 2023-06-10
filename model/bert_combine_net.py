import torch as pt
from torch import nn

from .bert_block import BertBlock


class BertCombineNet(nn.Module):
    def __init__(self, bert_dims=5, out_dims=2):
        super(BertCombineNet, self).__init__()

        self.title_bert = BertBlock(output_dims=bert_dims)
        self.abstract_bert = BertBlock(output_dims=bert_dims)
        self.fcs = nn.Sequential(
            nn.Linear(2 * bert_dims, 2 * bert_dims),
            nn.ReLU(),
            nn.Linear(2 * bert_dims, 2 * bert_dims),
            nn.ReLU(),
            nn.Linear(2 * bert_dims, out_dims)

        )

    def forward(self, pub_input):

        title_output = self.title_bert(*pub_input['title'])
        abstract_output = self.abstract_bert(*pub_input['abstract'])

        # now we can reshape `c` and `f` to 2D and concat them
        combined = pt.cat((title_output.view(title_output.size(0), -1),  abstract_output.view(abstract_output.size(0), -1)), dim=1)
        out = self.fcs(combined)

        return out
