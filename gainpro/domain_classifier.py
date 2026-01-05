import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    """ Distinguish the domain of the input.
    """

    def __init__(self, input_dim: int, n_class: int):
        super(DomainClassifier, self).__init__()

        # in the end is a logistic regressor
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_class)
        )

    def forward(self, x):
        return self.domain_classifier(x)