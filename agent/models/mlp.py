
import torch
from typing import List, Tuple

class WorldMlp(torch.nn.Module):
    '''
    General multi-layers perceptron model which can be used in
    world model
    '''

    def __init__(self,
        input_size: int,
        hidden_sizes: List,
        output_size: int,
        activation: torch.nn.Module=torch.nn.LeakyReLU,
        output_layer=None,
        use_batchnorm: bool=False,
        dropout_prob: float=None):
        super().__init__()

        layers_dim = [input_size] + hidden_sizes
        layers = []
        for i in range(len(layers_dim)-1):
            layers.append(torch.nn.Linear(layers_dim[i], layers_dim[i+1]))
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(layers_dim[i+1]))
            layers.append(activation())
            if dropout_prob is not None:
                layers.append(torch.nn.Dropout(dropout_prob))
        layers.append(torch.nn.Linear(layers_dim[-1], output_size))
        if output_layer:
            layers.append(output_layer)
        self._model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Overload forward function for torch.nn.Moduel
        '''
        return self._model(x)



