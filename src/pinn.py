
import numpy as np
import torch
import torch.nn as nn


class PINNNet(nn.Module):
    def __init__(self, n_state, t_scale, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, n_state)]
        self.net = nn.Sequential(*layers)
        self.t_scale = t_scale

    def forward(self, t):
        return self.net(t / self.t_scale)


class LorenzParams(nn.Module):
    def __init__(self, init=(8.0, 20.0, 2.0)):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(float(init[0])))
        self.rho = nn.Parameter(torch.tensor(float(init[1])))
        self.beta = nn.Parameter(torch.tensor(float(init[2])))

    def rhs(self, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return torch.stack([dx1, dx2, dx3], dim=1)

    def current_values(self):
        return dict(sigma=self.sigma.item(), rho=self.rho.item(), beta=self.beta.item())


class VdpParams(nn.Module):
    def __init__(self, init=(1.0,)):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(init[0])))

    def rhs(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = x2
        dx2 = self.mu * (1 - x1 ** 2) * x2 - x1
        return torch.stack([dx1, dx2], dim=1)

    def current_values(self):
        return dict(mu=self.mu.item())


class DuffingParams(nn.Module):
    def __init__(self, omega, init=(0.15, -0.5, 0.5, 0.2)):
        super().__init__()
        self.omega = omega
        self.delta = nn.Parameter(torch.tensor(float(init[0])))
        self.alpha = nn.Parameter(torch.tensor(float(init[1])))
        self.beta = nn.Parameter(torch.tensor(float(init[2])))
        self.gamma = nn.Parameter(torch.tensor(float(init[3])))

    def rhs(self, x):
        x1, x2, c, s = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        dx1 = x2
        dx2 = -self.delta * x2 - self.alpha * x1 - self.beta * x1 ** 3 + self.gamma * c
        dc = -self.omega * s
        ds = self.omega * c
        return torch.stack([dx1, dx2, dc, ds], dim=1)

    def current_values(self):
        return dict(delta=self.delta.item(), alpha=self.alpha.item(),
                    beta=self.beta.item(), gamma=self.gamma.item())


PARAM_MODELS = {"lorenz": LorenzParams, "vanderpol": VdpParams, "duffing": DuffingParams}


def train_pinn(t_obs, x_obs, system_name, n_state, t_span, omega=None,
                n_collocation=500, n_iters=3000, lr=1e-3,
                lambda_phys=1.0, verbose_every=500):
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)
    x_obs_t = torch.tensor(x_obs, dtype=torch.float32)
    t_scale = t_span[1] - t_span[0]
    t_colloc = torch.linspace(t_span[0], t_span[1], n_collocation).reshape(-1, 1)
    t_colloc.requires_grad_(True)

    net = PINNNet(n_state, t_scale)
    if system_name == "duffing":
        params = DuffingParams(omega=omega)
    else:
        params = PARAM_MODELS[system_name]()

    optimizer = torch.optim.Adam(list(net.parameters()) + list(params.parameters()), lr=lr)

    for it in range(n_iters):
        optimizer.zero_grad()

        # data loss
        x_pred_data = net(t_obs_t)
        data_loss = ((x_pred_data - x_obs_t) ** 2).mean()

        # physics loss
        x_pred_colloc = net(t_colloc)
        dxdt = torch.zeros_like(x_pred_colloc)
        for i in range(n_state):
            grad_outputs = torch.ones(x_pred_colloc.shape[0])
            dxdt[:, i] = torch.autograd.grad(
                x_pred_colloc[:, i], t_colloc, grad_outputs=grad_outputs, create_graph=True
            )[0].squeeze()
        rhs = params.rhs(x_pred_colloc)
        physics_loss = ((dxdt - rhs) ** 2).mean()

        loss = data_loss + lambda_phys * physics_loss
        loss.backward()
        optimizer.step()

        if verbose_every and it % verbose_every == 0:
            print(f"  iter {it:5d}  data_loss={data_loss.item():.5f}  "
                  f"physics_loss={physics_loss.item():.5f}")

    return net, params


def predict(net, t):
    t_t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        return net(t_t).numpy()


def score_params(params, true_values, tol=1e-6):
    current = params.current_values()
    errors = {
        k: abs(current[k] - true_values[k]) / (abs(true_values[k]) + tol)
        for k in true_values
    }
    return dict(current=current, true=true_values, rel_errors=errors,
                mean_rel_error=float(np.mean(list(errors.values()))))
