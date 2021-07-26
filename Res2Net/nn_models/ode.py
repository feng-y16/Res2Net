import torch
import numpy as np
from torchdiffeq import odeint_adjoint as ode_int
from .mlp import FirstOrderAutoEncoder, SecondOrderAutoEncoder
import pdb


class FirstOrderODENet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, learn_rate, non_linearity='tanh', tol=1e-6):
        super(FirstOrderODENet, self).__init__()
        self.ode_rhs = FirstOrderAutoEncoder(input_dim, hidden_dim, non_linearity)
        self.integration_time = torch.tensor([0, 1]).float()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=1e-5)
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.to(x.device)
        x = ode_int(self.ode_rhs, x, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        return x

    def forward_train(self, x, targets, train=True, return_scalar=True):
        if train:
            x = self.forward(x)
            loss = ((x - targets) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                x = self.forward(x)
        if return_scalar:
            return ((x - targets) ** 2).mean()
        else:
            return ((x - targets) ** 2).mean(dim=1)


class SecondOrderODENet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, learn_rate, non_linearity='tanh', tol=1e-6):
        super(SecondOrderODENet, self).__init__()
        self.ode_rhs = SecondOrderAutoEncoder(input_dim, hidden_dim, non_linearity)
        self.integration_time = torch.tensor([0, 1]).float()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=1e-5)
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.to(x.device)
        x = ode_int(self.ode_rhs, x, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        return x

    def forward_train(self, x, targets, train=True, return_scalar=True):
        if train:
            x = self.forward(x)
            loss = ((x - targets) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                x = self.forward(x)
        if return_scalar:
            return ((x - targets) ** 2).mean()
        else:
            return ((x - targets) ** 2).mean(dim=1)
