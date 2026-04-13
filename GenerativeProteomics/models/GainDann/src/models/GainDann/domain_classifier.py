import torch.nn as nn


class DomainClassifier(nn.Module):
    """ Distinguish the domain of the input.
    """
    def __init__(
        cls, 
        input_dim: int, 
        hidden_dim: int, 
        n_class: int
    ):
        super(DomainClassifier, cls).__init__()

        # in the end is a logistic regressor
        cls.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )

        cls.init_weights()
    
    def init_weights(cls) -> None:
        for m in cls.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(cls, x):
        return cls.domain_classifier(x)