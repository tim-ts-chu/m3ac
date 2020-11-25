
import torch
from typing import List, Tuple

class MLP(torch.nn.Module):
    '''
    General multi-layers perceptron model which can be used in both
    policy network and Q-network.
    '''

    def __init__(self,
        input_size: int,
        hidden_sizes: List,
        output_size: int,
        activation: torch.nn.Module=torch.nn.ReLU,
        output_layer=None):
        super().__init__()

        layers_dim = [input_size] + hidden_sizes
        layers = []
        for i in range(len(layers_dim)-1):
            layers.append(torch.nn.Linear(layers_dim[i], layers_dim[i+1]))
            layers.append(activation())
        layers.append(torch.nn.Linear(layers_dim[-1], output_size))
        if output_layer:
            layers.append(output_layer)
        self._model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Overload forward function for torch.nn.Moduel
        '''
        return self._model(x)

class PolicyNetwork(torch.nn.Module):
    '''
    Policy network uses MLP as underlying network and it take
    environment state as input ant output an action.
    '''

    def __init__(self,
            state_size: int,
            action_size: int,
            hidden_sizes: List):
        super().__init__()

        self._action_size = action_size

        self._model = MLP(
                input_size=state_size,
                hidden_sizes=hidden_sizes,
                output_size=action_size*2,
                activation=torch.nn.ReLU)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Overload forward function for torch.nn.Moduel
        '''
        out = self._model(state)
        mu = out[:, :self._action_size]
        log_std = out[:, self._action_size:]
        return mu, log_std

class QNetwork(torch.nn.Module):
    '''
    Q-network uses MLP as underlying network and it take
    environment state and action as input and output a estimated q-value.
    '''

    def __init__(self,
            state_size: int,
            action_size: int,
            hidden_sizes: List):
        super().__init__()

        self._model = MLP(
                input_size=state_size+action_size,
                hidden_sizes=hidden_sizes,
                output_size=1,
                activation=torch.nn.ReLU)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Overload forward function for torch.nn.Moduel
        '''
        x = torch.cat((state, action), dim=1)
        return self._model(x)

