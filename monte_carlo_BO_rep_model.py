import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as tt
from pytensor.compile.ops import as_op

import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize, normalize

#--------------------- ODE definition ---------------------------------------------------------

# Differential equations for the repressilator system
def repressilator(y, t, alpha, beta, alpha0, degp, n):
    M1, P1, M2, P2, M3, P3 = y
    dM1_dt = -M1 + alpha / (1 + P3**n) + alpha0
    dP1_dt = -beta * (degp * P1 - M1)
    dM2_dt = -M2 + alpha / (1 + P1**n) + alpha0
    dP2_dt = -beta * (degp * P2 - M2)
    dM3_dt = -M3 + alpha / (1 + P2**n) + alpha0
    dP3_dt = -beta * (degp * P3 - M3)
    return [dM1_dt, dP1_dt, dM2_dt, dP2_dt, dM3_dt, dP3_dt]
'''
# Define Theano operation for solving the ODE
@as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(alpha, beta):
    solution = odeint(repressilator, y0, t, args=(alpha, beta, alpha0, degp, n))
    return np.array(solution)
'''

#------------------ Uncertain parameters sampling using Monte-Carlo ----------------------------
# PyMC3 model
def monte_carlo_sampling(n_samples, meanlist,stdlist):
    #n_samples = 100  # Reduced number of samples for faster execution
    with pm.Model() as model:
        # Stochastic variables for alpha and beta
        alpha = pm.Normal('alpha', mu=meanlist[0], sigma=stdlist[0])
        beta = pm.Normal('beta', mu=meanlist[1], sigma=stdlist[1])

        # Forward model
        #  should I keep it?
        #solution = th_forward_model(alpha, beta)

        # Monte Carlo sampling
        trace = pm.sample(n_samples, tune=50, return_inferencedata=False)

    # Extract sampled parameter values
    alpha_samples = trace['alpha']
    beta_samples = trace['beta']

    return torch

#----------------- Bayesian optimisation using BoTorch ------------------------------------------

#defining the target oscillation profile


def oscillator(t, period=2 * torch.pi, threshold=0):
    t_normalized = 2 * torch.pi * (t % period) / period
    
    # Compute the sine wave
    sine_wave = torch.sin(t_normalized)
    
    # Apply threshold to get on-off states
    on_off_states = (sine_wave > threshold).float() # Convert boolean tensor to float (0.0 or 1.0)
    
    return on_off_states


#return loss as difference between the solved model and the objective
def objective_function(sol, obje):
    
    return loss

def bo_run(bounds, num_initial_points, number_of_iterations, uncertain_parameters):
    #Sobol sampling for initial space

    sobol = SobolQMCNormalSampler(num_samples=num_initial_points)
    initial_x = sobol(bounds)

    # Normalize initial_x to [0, 1] because BoTorch operates in the normalized space
    train_x = normalize(initial_x, bounds)

    # Initialize train_y vector for results
    train_y = torch.zeros(100)

    #define the sampled uncertain parameters
    alpha,beta = uncertain_parameters

    # Solve the ODE for each set of parameters
    for i in range(num_initial_points):
        tempparameters=unnormalize(train_x, bounds)
        alpha0, degp, n = tempparam[i,:]
        sol = odeint(repressilator, y0, t, args=(alpha, beta, alpha0, degp, n))
        train_y[i]= objective_function(sol, osci)

    
    # Define and fit GP model
    gp_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)



    # Optimization loop
    for _ in range(number_of_iterations):  # Number of iterations
        # Define the acquisition function
        EI = ExpectedImprovement(model=gp_model, best_f=train_y.max())

        # Optimize the acquisition function to find the new candidate
        new_x, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        lten=new_x.shape[0]

        new_y= torch.zeros(lten)
        for i in range(lten):
            tempparameters=unnormalize(new_x, bounds)
            alpha0, degp, n = tempparam[i,:]
            sol = odeint(repressilator, y0, t, args=(alpha, beta, alpha0, degp, n))
            new_y[i]= objective_function(sol, osci)

        # Update training points
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        # Update the model
        gp_model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)

    return train_x, train_y, ind, mll


def main():
    # Initial conditions
    M1_0, P1_0, M2_0, P2_0, M3_0, P3_0 = 0, 1, 0, 2, 0, 3
    y0 = [M1_0, P1_0, M2_0, P2_0, M3_0, P3_0]

    # Time points
    t = np.linspace(0, total_time, 1000)

    global y0,t
    # Define the bounds of the search space, these are minx_1,min_x_2,... and the second one is maxx_1, maxx_2
    bounds = torch.stack([torch.tensor([1.0, 0.001]), torch.tensor([3.0, 0.1])])
    num_initial_points = 100  # Number of initial points
    number_of_iterations=20

    meanlist=[1,10]
    stdlist=[0.1,1]
    
    alpha_samples,beta_samples=monte_carlo_sampling(100,meanlist,std_list)
    # sampled parameters and plot
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_samples,beta_samples, c="black",alpha=0.5)  # Plotting P1 for each sample

    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('MC samples')
    plt.show()

    ttorch = torch(t) # Generate 100 time values from 0 to 10
    osci=oscillator(ttorch)

    plt.figure(figsize=(10, 4))
    plt.plot(ttorch, osci)
    plt.ylim(-0.1, 1.1) # Adjust y-axis to clearly show on-off states
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Target Oscillator")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Starting Bayesian Optimization...")
    for number in len(alpha_samples):# Example way to define uncertain parameters
        x_samples, y_results, batch_index, fitted_model = bo_main(bounds, num_initial_points, number_of_iterations, [alpha_samples[number] beta_samples[number]])
        
        
        break
if __name__ == "__main__":
    main()

