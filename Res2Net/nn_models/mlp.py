# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np
import pdb


def choose_non_linearity(name):
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    else:
        raise ValueError("non-linearity not recognized")
    return nl


class MLP(torch.nn.Module):
    """Just a salt-of-the-earth MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, non_linearity='tanh'):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

        for layer in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(layer.weight)  # use a principled initialization

        self.non_linearity = choose_non_linearity(non_linearity)

    def forward(self, x):
        h = self.non_linearity(self.linear1(x))
        h = self.non_linearity(self.linear2(h))
        return self.linear3(h)


class MLPAutoEncoder(torch.nn.Module):
    """A salt-of-the-earth MLP AutoEncoder + some edgy res connections"""

    def __init__(self, input_dim, hidden_dim, output_dim, latent_dim=None, non_linearity='tanh'):
        super(MLPAutoEncoder, self).__init__()
        if latent_dim is None:
            latent_dim = hidden_dim
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, output_dim)

        for layer in [self.linear1, self.linear2, self.linear3, self.linear4,
                      self.linear5, self.linear6, self.linear7, self.linear8]:
            torch.nn.init.orthogonal_(layer.weight)  # use a principled initialization

        self.non_linearity = choose_non_linearity(non_linearity)

    def encode(self, x):
        h = self.non_linearity(self.linear1(x))
        h = h + self.non_linearity(self.linear2(h))
        h = h + self.non_linearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        h = self.non_linearity(self.linear5(z))
        h = h + self.non_linearity(self.linear6(h))
        h = h + self.non_linearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class FirstOrderAutoEncoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, non_linearity='tanh'):
        super(FirstOrderAutoEncoder, self).__init__()
        self.ae = MLPAutoEncoder(input_dim, hidden_dim, input_dim, non_linearity=non_linearity)

    def forward(self, t, x):
        return self.ae(x)


class SecondOrderAutoEncoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, non_linearity='tanh'):
        super(SecondOrderAutoEncoder, self).__init__()
        assert input_dim % 2 == 0
        self.ae = MLPAutoEncoder(input_dim, hidden_dim, input_dim // 2, non_linearity=non_linearity)
        self.alpha = torch.nn.Parameter(-torch.randn(1, input_dim // 2))
        self.beta = torch.nn.Parameter(-torch.randn(1, input_dim // 2))

    def forward(self, t, x):
        size = x.size()
        x_1 = x[:, 0:size[1] // 2]
        x_2 = x[:, size[1] // 2:]
        return torch.cat((x_2, self.ae(x) - self.alpha * x_2 - self.beta * x_1), dim=1)
