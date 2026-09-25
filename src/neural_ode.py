
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):

    def __init__(self, n_state, mean, std, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_state),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.zeros_(m.bias)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, t, x):
        y = (x - self.mean) / self.std         
        dy = self.net(y)                          
        dx = dy * self.std                        
        return dx


def train_neural_ode(t, x, n_state, window_time=0.25, batch_size=32,
                      n_iters=1500, lr=1e-3, verbose_every=200):
   

    t_torch = torch.tensor(t, dtype=torch.float32)
    x_torch = torch.tensor(x, dtype=torch.float32)
    dt = t[1] - t[0]
    n_points = len(t)

    window = max(int(round(window_time / dt)), 2)
    window = min(window, n_points - 1) 

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0

    func = ODEFunc(n_state, mean, std)
    optimizer = torch.optim.Adam(func.parameters(), lr=lr)

    for it in range(n_iters):
        starts = np.random.randint(0, n_points - window, size=batch_size)
        batch_x0 = x_torch[starts]                                 
        batch_t = torch.linspace(0, (window - 1) * dt, window)    
        batch_true = torch.stack(
            [x_torch[s:s + window] for s in starts], dim=1
        )  

        optimizer.zero_grad()
        pred = odeint(func, batch_x0, batch_t, method="rk4")       
        loss.backward()
        optimizer.step()

        if verbose_every and it % verbose_every == 0:
            print(f"  iter {it:5d}  loss={loss.item():.5f}")

    return func


def save_model(func, path):
    torch.save(func.state_dict(), path)


def load_model(path, n_state, mean, std):
    func = ODEFunc(n_state, mean, std)
    func.load_state_dict(torch.load(path))
    func.eval()
    return func


def rollout(func, x0, t):
    t_torch = torch.tensor(t, dtype=torch.float32)
    x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = odeint(func, x0_torch, t_torch, method="rk4")
    return pred.squeeze(1).numpy()  # (T, n_state)


def evaluate_rollout(pred, true, short_horizon_frac=0.1):
    n = len(true)
    h = max(int(n * short_horizon_frac), 5)
    short_rmse = float(np.sqrt(np.mean((pred[:h] - true[:h]) ** 2)))

    true_scale = np.abs(true).max()
    blew_up = bool(
        np.isnan(pred).any() or np.isinf(pred).any()
        or np.abs(pred).max() > 10 * true_scale
    )
    return dict(short_rmse=short_rmse, blew_up=blew_up,
                max_abs_pred=float(np.abs(pred).max()))
