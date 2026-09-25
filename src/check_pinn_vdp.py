
from systems import VDP, simulate
from pinn import train_pinn, score_params

t, x = simulate(VDP)
true_values = dict(mu=VDP["params"]["mu"])

print("Training PINN on clean Van der Pol data, full domain, 15000 iters...")
net, params = train_pinn(
    t, x, system_name="vanderpol", n_state=2, t_span=VDP["t_span"],
    n_iters=15000, verbose_every=2000,
)

score = score_params(params, true_values)
print("\nRecovered:", score["current"])
print("True:     ", score["true"])
print("Relative errors:", score["rel_errors"])
