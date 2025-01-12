import torch
from torch import nn 
from torch import Tensor
from typing import Optional, Tuple, Literal
from jaxtyping import Float

class Constant(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.bias = nn.Parameter(torch.randn(out_dim))
    
    def forward(self, x: Tensor):
        
        a = self.bias.reshape([1 for _ in x.shape[:-2]] + [-1, 1]).expand(*x.shape[:-2], -1, x.shape[-1])
        return a

class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        use_bias: Optional[bool]=True,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        group: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.use_bias=use_bias
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections=set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.group = group
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers <= 0:
            if self.use_bias:
                layers.append(Constant(self.out_dim*self.group))
            else:
                layers.append(nn.Identity())
        elif self.num_layers == 1:
            if self.group == 1:
                layers.append(nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias))
            else:
                layers.append(nn.Conv1d(self.in_dim * self.group, self.out_dim * self.group, 1, groups=self.group, bias=self.use_bias))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    if self.group == 1:
                        layers.append(nn.Linear(self.in_dim, self.layer_width))
                    else:
                        layers.append(nn.Conv1d(self.in_dim * self.group, self.layer_width, 1, groups=self.group))
                elif i in self._skip_connections:
                    if self.group == 1:
                        layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                    else:
                        layers.append(nn.Conv1d(self.group * (self.layer_width + self.in_dim), self.group * self.layer_width, 1, groups=self.group))
                else:
                    if self.group == 1:
                        layers.append(nn.Linear(self.layer_width, self.layer_width))
                    else:
                        layers.append(nn.Conv1d(self.layer_width, self.layer_width, 1, groups=self.group))
            
            if self.group == 1:
                layers.append(nn.Linear(self.layer_width, self.out_dim))
            else:
                layers.append(nn.Conv1d(self.layer_width, self.out_dim * self.group, 1, groups=self.group))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        if self.group > 1:
            in_tensor = in_tensor.unsqueeze(-1)
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1 if self.group == 1 else -2)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        if self.group > 1:
            x = x[..., 0]
        return x
