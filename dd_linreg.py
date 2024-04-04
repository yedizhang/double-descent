import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class linreg(nn.Module):
    def __init__(self, in_dim=100, out_dim=1):
        super().__init__()
        self.in_out = nn.Linear(in_dim, out_dim, bias=False)
        self._init_weights()

    def forward(self, x):
        out = self.in_out(x)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)


def sweep(repeat=20):
    N_ = np.linspace(10, 200, 20)
    Eg_avg = np.zeros(N_.shape)
    for n, N in enumerate(N_):
        for i in range(repeat):
            _, Eg = train(int(N))
            Eg_avg[n] += Eg[-1]
    Eg_avg = Eg_avg / repeat
    print(Eg_avg)
    plt.scatter(N_, Eg_avg)
    plt.ylabel('Generalization error')
    plt.xlabel('# of training samples')
    plt.show()


def train(N, D=100, SNR=1.414, T=500, plot=False):
    """
    N: number of training samples
    D: input dimension
    SNR: signal-to-noise ratio
    T: number of epochs
    """
    w = np.random.normal(0, SNR, D)
    noise = np.random.normal(0, 1, N)
    x_train = np.random.multivariate_normal(np.zeros(D), np.eye(D)/D, N)
    y_train = x_train @ w + noise
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train[:, np.newaxis]).float()

    model = linreg(in_dim=D)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=4)

    Ls = np.zeros(T)
    Eg = np.zeros(T)
    for epoch in range(T):
        optimizer.zero_grad()
        w_hat = [param.data.cpu().detach().numpy() for param in model.parameters()]
        w_hat = w_hat[0]
        # forward
        out = model(x_train)
        loss = criterion(out, y_train)
        Ls[epoch] = loss.item()
        Eg[epoch] = np.linalg.norm(w_hat - w)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if plot:
        plt.plot(Ls/Ls[0], 'k') 
        plt.plot(Eg/Eg[0], 'r')
        plt.show()
    print("N={}, Loss={}, Eg={}".format(N, Ls[-1], Eg[-1]))

    return Ls, Eg


if __name__ == "__main__":
    train(N=100, D=100, T=500, plot=True)
    sweep()
