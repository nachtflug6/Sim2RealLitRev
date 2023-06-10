import torch as pt
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, n_title, n_abstract, n_hidden):
        super().__init__()

        self.title_mask_input = nn.Sequential(
            nn.Linear(n_title, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.title_id_input = nn.Sequential(
            nn.Linear(n_title, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.abstract_mask_input = nn.Sequential(
            nn.Linear(n_abstract, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.abstract_id_input = nn.Sequential(
            nn.Linear(n_abstract, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.model = nn.Sequential(
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, title_id, title_mask, abstract_id, abstract_mask):
        title_id_input = self.title_id_input(title_id)
        title_mask_input = self.title_mask_input(title_mask)
        abstract_id_input = self.abstract_id_input(abstract_id)
        abstract_mask_input = self.abstract_mask_input(abstract_mask)

        combined = pt.cat((title_id_input.view(title_id_input.size(0), -1),
                           title_mask_input.view(title_mask_input.size(0), -1),
                           abstract_id_input.view(abstract_id_input.size(0), -1),
                           abstract_mask_input.view(abstract_mask_input.size(0), -1)), dim=1)

        output = self.model(combined)

        return output


class Generator(nn.Module):

    def __init__(self, n_title, n_abstract, n_hidden):
        super().__init__()

        self.title_mask_input = nn.Sequential(
            nn.Linear(n_title, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.title_id_input = nn.Sequential(
            nn.Linear(n_title, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.abstract_mask_input = nn.Sequential(
            nn.Linear(n_abstract, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.abstract_id_input = nn.Sequential(
            nn.Linear(n_abstract, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.body = nn.Sequential(
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden),
            nn.ReLU(),
        )

        self.title_mask_output = nn.Sequential(
            nn.Linear(4 * n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_title),
        )

        self.title_id_output = nn.Sequential(
            nn.Linear(4 * n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_title),
        )

        self.abstract_mask_output = nn.Sequential(
            nn.Linear(4 * n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_abstract),
        )

        self.abstract_id_output = nn.Sequential(
            nn.Linear(4 * n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_abstract),
        )

    def forward(self, title_id, title_mask, abstract_id, abstract_mask):
        title_id_input = self.title_id_input(title_id)
        title_mask_input = self.title_mask_input(title_mask)
        abstract_id_input = self.abstract_id_input(abstract_id)
        abstract_mask_input = self.abstract_mask_input(abstract_mask)

        combined = pt.cat((title_id_input.view(title_id_input.size(0), -1),
                           title_mask_input.view(title_mask_input.size(0), -1),
                           abstract_id_input.view(abstract_id_input.size(0), -1),
                           abstract_mask_input.view(abstract_mask_input.size(0), -1)), dim=1)

        title_id_output = self.title_id_output(combined)
        title_mask_output = self.title_mask_output(combined)
        abstract_id_output = self.abstract_id_output(combined)
        abstract_mask_output = self.abstract_mask_output_mask_output(combined)

        return title_id_output, title_mask_output, abstract_id_output, abstract_mask_output
