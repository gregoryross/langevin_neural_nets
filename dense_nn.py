import jax.scipy.stats as stats
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.flatten_util import ravel_pytree
#from jax.experimental import optimizers
from jax.example_libraries.optimizers import adam
from jax.scipy.special import expit

import itertools

# Non linear functions taken from jax.experimental.stax
def relu(x): return np.maximum(x, 0.)
def softplus(x): return np.logaddexp(x, 0.)
def sigmoid(x): return expit(x)
def elu(x): return np.where(x > 0, x, np.expm1(x))
def leaky_relu(x): return np.where(x >= 0, x, 0.01 * x)


def softmax(x, axis=-1):
    """Apply softmax to an array of logits, exponentiating and normalizing along an axis."""
    unnormalized = np.exp(x - x.max(axis, keepdims=True))
    return unnormalized / unnormalized.sum(axis, keepdims=True)


def positanh(x):
    """
    Does not have odd symmetry about x=0 (i.e. tanh(-x) = -tanh(x)) but maintains the same performance as tanh.
    """
    return 1. + np.tanh(x)


def baoab(dt, beta=1., mass=1., gamma=1.):
    """
    A langevin integrator with the BAOAB splitting.
    """
    zeta = ((1. - np.exp(- 2 * gamma * dt)) / beta) ** 0.5
    expon = np.exp(- dt * gamma)

    def integrator(g, q, p, key):
        """
        Taken a single timestep with a the BAOAB langevin integrator.

        Parameters
        ----------
        g: jax.interpreters.xla.DeviceArray
            The gradient of each of the unknown variables. Same size as q.
        q: jax.interpreters.xla.DeviceArray
            The current values of the unknown variables.
        p: jax.interpreters.xla.DeviceArray
            The fictitious momenta of the unknown variables.
        key: jax.interpreters.xla.DeviceArray
            The key for the generation of random numbers (key.size should be 2).

        Returns
        -------
        q, p: the updated variables and momenta.
        """
        p = p - g * dt  # No factor of 2 to accounts for the velocity kicks at the start and end.
        q = q + (dt / 2.) * p / mass
        p = expon * p + zeta * np.sqrt(mass) * random.normal(key, p.shape)
        q = q + (dt / 2.) * p / mass
        return q, p

    return integrator


def init_nn_params(layer_shapes, scale, seed):
    """
    Build a list of (weights, biases) tuples, one for each layer. Adapted from JAX. Each parameter is drawn from a
    Guassian with a mean of zero and a standard deviation given by the scale parameter.

    TODO
    ----
    Scale the prior width between each layer by the square-root of the number of units next that layer.
    This ensures each data point has a well defined prior output from each layer.
    See Radford Neal's thesis "Bayesian Learning for Neural Networks" which discusses this for the last hidden to output layer.

    Parameters
    ----------
    layer_shapes: tuple/list of ints
        The number of units in each layer. The first entry and last entry must much the dimensionality of the inputs and
        outputs. E.g. later_shapes = (1,50,50,1) implies a NN with 1 input, 2 hidden layers each of 50 units, and one
        output.
    scale: float
        The standard deviation of the Gaussian from which _all_ weights and biases are drawn from.
    seed: int
        The random seed used to generate the parameters.

    Returns
    -------
    params: list
        The parameters of the NN. Must be compatiple with nn_predict.
    """
    key = random.PRNGKey(seed)
    split_keys = random.split(key, 2 * len(layer_shapes))
    return [(random.normal(split_keys[i], (size[0], size[1])) * scale,  # weight matrix
             random.normal(split_keys[-(i + 1)], (size[1],)) * scale)  # bias vector
            for i, size in enumerate(zip(layer_shapes[:-1], layer_shapes[1:]))]


def nn_predict(params, inputs, nonlinearity=np.tanh):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs


class GaussianDenseNN(object):
    """
    A Bayesian Dense neural network which can be sampled using Langevin dynamics.
    """

    def __init__(self, inputs, targets, layer_shapes, prior_weight_scale=1, prior_error_scale=1, nonlinear=positanh,
                 seed=0):
        """
        Class for a Bayesian dense neural network with Gaussian priors and likelihood.
        """
        # training data
        self.inputs = inputs
        self.targets = targets

        # NN meta params
        # --------------
        self.layer_shapes = layer_shapes
        self.nonlinear = nonlinear
        self.seed = seed
        self.mass = 10  # The mass of each weight and bias. Used for
        self.beta = 1
        # MCMC statistics
        # ---------------
        self.nsteps = 0

        # Prior info
        # ----------
        self.prior_param_scale = prior_weight_scale
        self.prior_error_scale = prior_error_scale

        # Initialize parameters
        # ---------------------
        self.key = random.PRNGKey(self.seed)
        # Setting the standard deviation to be small irrespective of the prior.
        self.params = init_nn_params(self.layer_shapes, scale=self.prior_param_scale, seed=self.seed) ## The weights and biases for the NN
        # Define the raveling function. This packs a 1D array of NN params into the shape required for prediction.
        unravelled_params, self.ravel = ravel_pytree(self.params)
        # The scale of the loss error
        error_scale = random.gamma(self.key, self.prior_error_scale, (1,))

        # The parameters that will be updated
        self.unknowns = np.hstack((error_scale, unravelled_params))
        self.momenta = self.init_random_momenta(self.unknowns)

        # Integrator tools
        # ----------------
        self.gamma = 100.
        self.dt = 0.0001
        self.integrator = baoab(dt=self.dt, beta=self.beta, mass=self.mass, gamma=self.gamma)

        # The energy and gradient function
        self.energy, self.gradient, self.log_like, self.log_prior = self.init_energy_and_grad()

        # Coupling parameter for the likelihood.
        self.lam = 1.0

    def init_random_momenta(self, unravelled_params, seed=None):
        if seed is None:
            self.seed += 1
        key = random.PRNGKey(self.seed)
        return random.normal(key, unravelled_params.shape) * self.beta / self.mass

    def predict(self):
        # TODO: remove the last nonlinearity to save computation.
        unravelled_nn_params = self.unknowns[1:]
        params = self.ravel(unravelled_nn_params)
        return nn_predict(params, self.inputs, nonlinearity=self.nonlinear)

    def reinit_integrator(self, dt=None, beta=None, mass=None, gamma=None):
        """
        Re-initialize the integrator. Momenta will be re-drawn with a new random seed.
        """
        if beta is None:
            beta = self.beta
        if mass is None:
            mass = self.mass
        if gamma is None:
            gamma = self.gamma
        if dt is None:
            dt = self.dt

        self.integrator = baoab(dt=dt, beta=beta, mass=mass, gamma=gamma)
        self.nsteps = 0

    def batch_log_like(self, unknowns, new_inputs, new_targets, lam=1.):
        """
        Compute the log-likelihood for any data given the model.

        TODO: Add student's t-distribution to give greater flexibility.
        TODO: E.g. would allow the annealing of the degree of freedom parameter from 1 to infinity.
        """
        params = self.ravel(unknowns[1:])
        pred = nn_predict(params, new_inputs, nonlinearity=self.nonlinear)
        return lam * np.sum(stats.norm.logpdf((pred - new_targets).flatten(), scale=unknowns[0]))

    def init_energy_and_grad(self):
        """
        Factory function for energy and gradient.
        """

        def log_like(unknowns):
            return self.batch_log_like(unknowns, self.inputs, self.targets, self.lam)

        def log_prior(unknowns):
            # TODO: scale the weights by the sqrt of number hidden units
            weight_prior = np.sum(stats.norm.logpdf((unknowns[1:]).flatten(), scale=self.prior_param_scale))
            error_prior = stats.gamma.logpdf(unknowns[0], a=self.prior_error_scale)
            return error_prior + weight_prior

        def energy(unknowns):
            return -log_like(unknowns) - log_prior(unknowns)

        gradient = jit(grad(energy))

        return energy, gradient, log_like, log_prior

    def reinit_system(self):
        """
        Re-initialize the log-posterior (energy), gradient, log likelihood, and log-prior after any changes to the
        inputs and targets.
        """
        self.energy, self.gradient, self.log_like, self.log_prior = self.init_energy_and_grad()

    def step(self, key):
        """
        Update the parameters by a single time-step. A key for the random number generator is required.
        """
        g = self.gradient(self.unknowns)
        self.unknowns, self.momenta = self.integrator(g, self.unknowns, self.momenta, key)
        self.nsteps += 1

    def get_maxlike(self, step_size=0.001, nsteps=1000, loud=True):
        """
        Use adam to minimize the sum of squares.
        """

        def square_loss(nn_params):
            pred = nn_predict(nn_params, self.inputs, nonlinearity=self.nonlinear)
            return np.sum((pred - self.targets) ** 2)

        loss_gradient_fun = jit(grad(square_loss))

        opt_init, opt_update, get_params = adam(step_size)

        def update(i, opt_state):
            params = get_params(opt_state)
            return opt_update(i, loss_gradient_fun(params), opt_state)

        params = self.ravel(self.unknowns[1:])
        opt_state = opt_init(params)
        itercount = itertools.count()

        # If noisy, print the energy after every 10% of the minimizer.
        percent = 10.
        completion_freq = int(nsteps / percent)

        k = 0
        for i in range(nsteps):
            opt_state = update(next(itercount), opt_state)
            if loud:
                if i % completion_freq == 0:
                    params = get_params(opt_state)
                    print("{0}% completed. Loss = {1:1.2f}".format(k * percent, square_loss(params)))
                    k += 1
        if loud:
            print('100% completed. Energy = {0:1.2f}'.format(square_loss(params)))

        self.params = params
        unravelled_params, _ = ravel_pytree(params)
        self.unknowns = np.hstack((self.unknowns[0], unravelled_params))

    def get_map(self, step_size=0.001, nsteps=1000, loud=True):
        """
        Use the adam optimizer to update the parameters to their maximum aposteriori values.
        """
        # If noisy, print the energy after every 10% of the minimizer.
        percent = 10.
        completion_freq = int(nsteps / percent)

        opt_init, opt_update, get_params = adam(step_size)

        def update(i, opt_state):
            all_params = get_params(opt_state)
            return opt_update(i, self.gradient(all_params), opt_state)

        opt_state = opt_init(self.unknowns)
        itercount = itertools.count()

        k = 0
        for i in range(nsteps):
            opt_state = update(next(itercount), opt_state)
            if loud:
                if i % completion_freq == 0:
                    all_params = get_params(opt_state)
                    print("{0}% completed. Energy = {1:1.2f}".format(k * percent, self.energy(all_params)))
                    k += 1
        if loud:
            print('100% completed. Energy = {0:1.2f}'.format(self.energy(all_params)))
        self.unknowns = get_params(opt_state)
