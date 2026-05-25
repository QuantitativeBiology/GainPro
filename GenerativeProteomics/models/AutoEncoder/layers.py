import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        use_batch_norm: bool=True,
        activation: type[nn.Module]=nn.LeakyReLU,
        output_activation: type[nn.Module] | None=None,
    ) -> "MLP":
        super().__init__()

        layers: list[nn.Module] = []

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            if i < len(layer_sizes) - 1:
                if use_batch_norm:
                    layers.append(
                        nn.BatchNorm1d(
                            layer_sizes[i]
                        )
                    )
                layers.append(activation())
            else:
                if output_activation is not None:
                    layers.append(output_activation())

        self.net = nn.Sequential(*layers)
        self.initialize_weights()
    
    def initialize_weights(
        self
    ) -> None:
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]

        for module in self.net:
            if isinstance(module, nn.Linear):
                is_output = (module is linear_layers[-1])

                if is_output and isinstance(module, (nn.Tanh, nn.Sigmoid)):
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(
                        module.weight,
                        a=0.01,
                        nonlinearity="leaky_relu" if isinstance(module, nn.LeakyReLU) else "relu"
                    )
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)