# Langevin neural nets
For performing langevin dynamics and sequential Monte Carlo on neural networks. This repo serves as a Bayesian neural net
sandbox for prototyping sampling ideas. 

The core sampling methodologies are the [BAOAB langevin intetegrator](https://doi.org/10.1098/rspa.2016.0138), [SMC with
data annealing](https://doi.org/10.1023/A:1024084221803), and SMC with geometric annealing of the likelihood. Both SMC
methods also use langevin dynamics as the MCMC kernel. 

## Manifest
Files:
* `dense_nn.py`: A set of tools and a class for performing Bayesian inference with neural nets. The focus is on
regression problems.
* `smc_samplers.py`: A set of methods for running sequential Monte Carlo on Bayesian neural nets.
* `sci_toopls.py`: General purpose tools for aiding the validation of the Bayesian neural nets.
* `test_bayes_nns.py`, `test_smc_samplers.py`: unit tests for the above.

Directories:
* `examples/`: A collection of working examples that are for demonstration and validation of the sampling methodology.
* `notebooks/`: Jupyter notebooks for the testing and running of the code. 

## Usage
### Quick set-up of a regression problem
```python
from dense_nn import GaussianDenseNN  
from jax import random
import jax.numpy as np
```
First, we'll generate a toy regression problem and then perform regression using Bayesian neural nets. Estimation will
involve MCMC sampling of the weights using Langevin dynamics.

Our repsonse, or target variable will be a 1D curve with Gaussian white noise. As Bayesian methods are well suited to
small data problems, we'll have only 50 data points.
```python
INPUTS = np.linspace(0,1, N).reshape((N,1))
def true_response(x):
    return 4*np.sin(x - 2) + np.sin(8*x - 2 ) - 4*x**2
TARGET =  true_response(INPUTS) + random.normal(key,(N,1)) * 0.5
```
The goal in this example is to try and provide decent credible intervals for the "true" relationship between `INPUTS`
and `TARGET`.
### MCMC sampling of Bayesian neural nets
Initializing the Bayesian neural net. We'll use 2 hidden layers with 50 units each. The parametets `prior_weight_scale`
and `prior_error_scale` set the standard deviation of the Gaussian prior and gamma distribution parameter for the 
prior on the data error.
```python
nn = GaussianDenseNN(INPUTS, TARGET, (1,50,50,1), prior_weight_scale=1, prior_error_scale=1, seed=0)
```
An estimate for the relationship between the functional form can be arrived at using the maximum a posteriori
estimate (MAP):
```python
nn.get_map(nsteps=1000)
```
A more "Bayesian" approach is to sample the weights of the nueral nets with MCMC. The MAP estimate will be used for our
starting point for a burn-in phase, followed by a production phase where we'll collect the prediction samples. For the 
burn-in, we'll use a high coefficient of friction (`gamma`) and large mass (`mass`) in the Langevin integration.

#### The burn-in
```python
nn.reinit_integrator(dt=0.001, beta=1, mass=10, gamma=100)
nn.seed += 1
key = random.PRNGKey(nn.seed)
split_keys = random.split(key, nsteps)
nn.seed += 1
for i in range(nsteps):
    nn.step(key=split_keys[i])
```
#### The production sampling
Here, we'll decrease the mass and coefficient of friction to get more efficient sampling. In the below, we'll only be recording the
predictions, but we can also track the weights by storing `nn.unknowns` for each step. 
```python
nsteps = 100000
completion_freq = int(nsteps/10)

nn.reinit_integrator(dt=0.001, beta=1, mass=1, gamma=10)
key = random.PRNGKey(nn.seed)
split_keys = random.split(key, nsteps)
nn.seed += 1

predictions = []

k=0
for i in range(nsteps):
    nn.step(key=split_keys[i])
    if i % 100 == 0:
        predictions.append(nn.predict())
    if i % int(nsteps/10) == 0:
        print('{0}% completed.'.format(k*percent))
        k += 1
print('100% completed.')
```
The posterior estimates of the relationship between `INPUTS` and `TARGET` are stored in the list `predictions`.

### SMC sampling of Bayesian neural nets
This repo also has experimental sequential Monte Carlo (SMC) samplers. These have a great deal of potential as they can be
easily parallelised. The SMC methods explored here use an ensemble of 
Bayesian neural nets to sample from. Each neural net has a Langevin MCMC sampling kernel.  

#### Annealing the data one-by-one
This method sequetially adds each data point one-by-one. After the addition of each data point, the neural nets are 
resampled and updated with `nsteps` of Langevin dynamics.
```python
import smc_samplers

nsteps = 100
nreplicas = 50
nn, replica_unknowns, sample_sizes, resample_times = smc_samplers.smc_data_addition(INPUTS, TARGET, nreplicas, 
                                                                       layer_shapes=(1,50,50,1), draw_param_scale=0.5, 
                                                                       draw_error_scale=0.5, ess_frac=0.7, 
                                                                       nsteps=nsteps, seed=0, loud=True)
```
The output contains a single neural net (`nn`) with a corresponding ensemble of weights (`replica_unkowns`). The neural
net can be updated with the `i`th member of the ensemble via `nn.unknowns = replica_unknowns[i]`.

The other SMC method this repo contains is one where the data is annealed geometrically. This is done by using a
 likelihood function that is set to the power of a parameter that lies between 0 and 1. With this parameter is 0, the
 data has no effect on the posterior, when this parameter equals 1, the posterior feels the full influence of the data. The
 SMC scheme works by gradually increasing this parameter from 0 to 1 a total of `nstages` times.  
```python
nn, replica_unknowns, resample_times = smc_samplers.smc_geometric_likelihood(INPUTS, TARGET, nreplicas=nreplicas,
                                                                            layer_shapes=(1,50,50,1), nstages=len(INPUTS), 
                                                                            draw_param_scale=0.5, draw_error_scale=0.5,
                                                                            ess_frac=0.7, nsteps=nsteps, seed=0, loud=True)
```