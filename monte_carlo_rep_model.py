import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as tt
from pytensor.compile.ops import as_op

# Parameters
n = 2       # Hill coefficient
total_time = 15  # total time for simulation
alpha0 = 0.01
degp = 0.1

# Initial conditions
M1_0, P1_0, M2_0, P2_0, M3_0, P3_0 = 0, 1, 0, 2, 0, 3
y0 = [M1_0, P1_0, M2_0, P2_0, M3_0, P3_0]

# Time points
t = np.linspace(0, total_time, 1000)

# Differential equations for the repressilator system
def repressilator(y, t, alpha, beta, n):
    M1, P1, M2, P2, M3, P3 = y
    dM1_dt = -M1 + alpha / (1 + P3**n) + alpha0
    dP1_dt = -beta * (degp * P1 - M1)
    dM2_dt = -M2 + alpha / (1 + P1**n) + alpha0
    dP2_dt = -beta * (degp * P2 - M2)
    dM3_dt = -M3 + alpha / (1 + P2**n) + alpha0
    dP3_dt = -beta * (degp * P3 - M3)
    return [dM1_dt, dP1_dt, dM2_dt, dP2_dt, dM3_dt, dP3_dt]

# Define Theano operation for solving the ODE
@as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(alpha, beta):
    solution = odeint(repressilator, y0, t, args=(alpha, beta, n))
    return np.array(solution)

# PyMC3 model
n_samples = 100  # Reduced number of samples for faster execution
with pm.Model() as model:
    # Stochastic variables for alpha and beta
    alpha = pm.Normal('alpha', mu=1, sigma=0.1)
    beta = pm.Normal('beta', mu=10, sigma=1)

    # Forward model
    solution = th_forward_model(alpha, beta)

    # Monte Carlo sampling
    trace = pm.sample(n_samples, tune=50, return_inferencedata=False)

# Extract sampled parameter values
alpha_samples = trace['alpha']
beta_samples = trace['beta']

# Solve ODE for sampled parameters and plot
plt.figure(figsize=(10, 6))
for i in range(len(alpha_samples)):
    sol = odeint(repressilator, y0, t, args=(alpha_samples[i], beta_samples[i], n))
    plt.plot(t, sol[:, 1], 'b', alpha=0.1)  # Plotting P1 for each sample

plt.xlabel('Time')
plt.ylabel('P1 concentration')
plt.title('Uncertainty in P1 Trajectories')
plt.show()

# Solve ODE for sampled parameters and plot
plt.figure(figsize=(10, 6))
plt.scatter(alpha_samples,beta_samples, c="black",alpha=0.5)  # Plotting P1 for each sample

plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('MC samples')
plt.show()