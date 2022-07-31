import jax.numpy as jnp
import jax
import numpy as np
import dense_nn

"""
Functions and classes to trial sequential Monte Carlo samplers.
"""

def gen_data_splits(ndata, packet_size):
    """
    Generate the start and end indices for dividing an array into equally sized packets. If the requested
    packet size does not divide equally into the number of data points, the final packet will be the reminder.

    Parameters
    ----------
    ndata: int
        The number of data points, i.e. the length of the array that will be divided.
    packet_size: int
        The length of data packets that will be created.

    Returns
    -------
    start_points, end_points: numpy.ndarray
        The indices for the start and end of the data packets. e.g. the nth data packet has indices between start_points[n]
        and end_points[n].
    """
    data_step = int(ndata / packet_size)
    start_points = np.arange(0, data_step) * packet_size
    end_points = start_points + packet_size

    if ndata % packet_size != 0:
        start_points = np.hstack((start_points, start_points[-1] + packet_size))
        end_points = np.hstack((end_points, ndata))

    return start_points, end_points


def calc_weights_sequential_data(nn, replica_unknowns, log_weights, new_inputs, new_targets):
    """
    Update the weights of the replicas/particles/ensemble for the SMC strategy that adds data to the likelihood.

    Parameters
    ----------
    nn: GaussianDenseNN
        The NN that can calculate the loglikelihood.
    replica_unknowns: list of numpy.ndarray
        The model parameters for each replica/particular that are being sampled over.
    log_weights: numpy.ndarray
        The current log weights of the replicas/particles.

    Return
    ------
    log_weights: numpy.ndarray
        The updated log weights.
    weights: numpy.ndarray
        The normalized weights.
    """
    for i in range(len(replica_unknowns)):
        log_weights[i] += float(nn.batch_log_like(replica_unknowns[i], new_inputs, new_targets, lam=1.))
        #log_weights[i] += float(nn.log_like(replica_unknowns[i]))
    # Stabalize weight calculation:
    weights = np.exp(log_weights - log_weights.max())
    weights = weights / np.sum(weights)
    return log_weights, weights


def calc_weights_geometric(nn, dlam, replica_unknowns, log_weights):
    """
    Update the weights for the sequential scheme where the likelihood is coupled in geometrically.
    """
    for i in range(len(replica_unknowns)):
        log_weights[i] += dlam * float(nn.batch_log_like(replica_unknowns[i],nn.inputs, nn.targets, lam=1))
    
    # Stabilise weight calculation:
    weights = np.exp(log_weights - log_weights.max())
    weights = weights / np.sum(weights)
    return log_weights, weights


def generate_replicas(nn, nreplicas, seed=0):
    """
    Create multiple replicas of the NN parameters using the same parameters as a single NN object.

    Parameters
    ----------
    nn: GaussianDenseNN
        The neural network on which the replicas/particles are based on.
    nreplicas: int
        The number of replicas/particles.
    seed: int
        The random seed.

    Returns
    -------
    replica_unknowns: list of numpy.ndarray
        Multiple replicas of variables drawn from the same prior as the supplied Gaussian NN.
    seed: int
        A new random seed that hasn't been used to generate the output data.
    """
    replica_unknowns = []
    split_keys_params = jax.random.split(jax.random.PRNGKey(seed), nreplicas)
    seed += 1
    split_keys_error = jax.random.split(jax.random.PRNGKey(seed), nreplicas)
    for i in range(nreplicas):
        param_array = jax.random.normal(split_keys_params[i], (nn.unknowns.shape[0] - 1,)) * nn.prior_param_scale
        error_scale = jax.random.gamma(split_keys_error[i], nn.prior_error_scale, (1,))
        replica_unknowns.append(jnp.hstack((error_scale, param_array)))

    return replica_unknowns, seed


def ess(weights):
    """
    Calculate the effective sample size of the weights.
    """
    return np.sum(weights) ** 2 / np.sum(weights ** 2)


def resample_replicas(weights, replica_unknowns):
    """
    Resample the replicas/particles via multinomial bootstrap sampling of the weights.

    Parameters
    ----------
    weights: numpy.ndarray
        The weights of the replicas.
    replica_unknowns: list of numpy.ndarray
        The model parameters for each replica/particular that are being sampled over.

    Returns
    -------
    replica_unknowns: list of numpy.ndarray
        The model parameters of the resampled replicas/particles.
    """
    nreplicas = len(replica_unknowns)
    inds = np.random.choice(np.arange(0, nreplicas), replace=True, p=weights, size=nreplicas)
    return [replica_unknowns[i] for i in inds]


def advance_replicas(replica_unknowns, nn, nsteps, seed):
    """
    Advance each of the replicas with a number of Langevin steps.

    Parameters
    ----------
    replica_unknowns: list of numpy.ndarray
        The model parameters for each replica/particular that will be propagated with langevin dynamics.
        These parameters will modified in place.
    nn: GaussianDenseNN
        The NN object that can propagate the parameters with Langevin dynamics.
    nsteps: int
        The number of steps of Langevin dynamics that will be performed.
    seed: int
        The random seed.

    Returns
    -------
    seed: int
        A random seed that has not been used in the current batch.
    """
    for i in range(len(replica_unknowns)):
        # Assign replica parameters to the NN
        nn.unknowns = replica_unknowns[i]
        # Re-draw momenta.
        nn.init_random_momenta(replica_unknowns[i], seed=seed)  # Automatically applies a new random seed.
        seed += 1
        # Perform multiple steps of Langevin dynamics
        key = jax.random.PRNGKey(seed)
        split_keys = jax.random.split(key, nsteps)
        for j in range(nsteps):
            nn.step(key=split_keys[j])
        # Record the final parameters.
        replica_unknowns[i] = nn.unknowns
        seed += 1

    return seed


def smc_data_addition(inputs, target, nreplicas, layer_shapes, draw_param_scale, draw_error_scale,
                      ess_frac=0.7, nsteps=100, seed=0, loud=True):
    """
    Function to perform sequential Monte Carlo fitting of a neural net in which the data is gradually
    added to the likelihood.

    As decribed by Ridgeway and Madigan in "A Sequential Monte Carlo Method for Bayesian Analysis of
    Massive Datasets", 2003, Data Min Knowl Discov. 301–319.

    Parameters
    ----------
    inputs: array like
        The explanatory variables, where each variable is in a different column.
    target: array like
        The response variable.
    nreplicas: int
        The number of SMC particles.
    layer_shapes: tuple of ints
        The number of input layers, hidden layers, and output layers of the
    draw_param_scale: float
        The stanadard deviation of the priors over the NN weights and biases. The priors have a mean of zero.
    draw_error_scale: float
        The scale parameter of the gamma prior on the standard deviation of the likelihood.
    ess_frac: float between 0 and 1.
        The minimum allowed fractional effective sample size. A value below this triggers resampling of the replicas
        and Langevin dynamics.
    nsteps: int
        The number of Langevin dynamics steps taken on each replica after resampling.
    seed: int
        The random seed.
    lout: bool
        Whether to occasionally print out status of the sampler.

    Returns
    -------
    nn: GaussianDenseNN
        The Bayesian nueral network that has been powering the sampling.
    replica_unknowns: list of jax arrays
        The parameters of the SMC replicas/particles. Can be slotted into the nn.
    sample_sizes: list
        A record of the effective sample sizes
    resample_times: list
        The times when resampling was triggered.
    """
    # Sanity checks:
    if layer_shapes[0] != inputs.shape[1]:
        raise Exception(
            'The number of input layers ({0}) should match the number of explanatory variables ({1})'.format(
                layer_shapes[0], inputs.shape[1]))
    if layer_shapes[-1] != target.shape[1]:
        raise Exception('The number of output layers ({0}) should match the number of response variables ({1})'.format(
            layer_shapes[-1], target.shape[1]))

    # Split the training data into packets
    packet_size = 1  # The number of data points added in SMC round
    start_points, end_points = gen_data_splits(inputs.shape[0], packet_size)
    nstages = len(start_points)
    stage = 0  # The current SMC stage

    # Initialize the NN with the initial data packet
    initial_inputs = jnp.array(inputs[start_points[stage]:end_points[stage], :])
    initial_target = jnp.array(target[start_points[stage]:end_points[stage], :])

    nn = dense_nn.GaussianDenseNN(initial_inputs, initial_target, layer_shapes,
                                  prior_weight_scale=draw_param_scale, prior_error_scale=draw_error_scale, seed=seed)
    nn.reinit_integrator(dt=0.0001, gamma=100, mass=10)
    seed += 1

    # Generate the particles
    replica_unknowns, seed = generate_replicas(nn, nreplicas, seed=seed)

    # Initialize the weights
    log_weights = np.zeros(nreplicas)  # Assumes the replicas have been drawn from the prior
    sample_sizes = np.zeros(nstages)
    resample_times = []

    # Print the status every this percent:
    percent = 10
    completion_freq = int(nstages * percent / 100)
    k = 1
    for n in range(1, nstages):
        if n != 1:
            # Add new data to nn:
            # Don't bother for the first data point because it's already loaded in the nn.
            nn.inputs = np.vstack((nn.inputs, inputs[start_points[n]:end_points[n]]))
            nn.targets = np.vstack((nn.targets, target[start_points[n]:end_points[n]]))
            nn.reinit_system()

        # Update the weights
        log_weights, weights = calc_weights_sequential_data(nn, replica_unknowns, log_weights,
                                                                         inputs[start_points[n]:end_points[n]],
                                                                         target[start_points[n]:end_points[n]])
        # Calculate the effective sample size
        ancestor_frac = np.sum(np.floor(weights * nreplicas) > 0) / nreplicas
        s = ess(weights)

        if n % 5 == 0:
            if loud:
                if n % completion_freq == 0:
                    print('{0}% completed. ESS = {1:.1f}, Progenitors={2:.1f}'.format(k * percent, s, ancestor_frac))
                    k += 1
        if s <= ess_frac * nreplicas:
            if loud:
                print('  Resampled at stage {0} of {1}'.format(n, nstages))
            resample_times.append(n)

            # Resample with replacement
            replica_unknowns = resample_replicas(weights, replica_unknowns)

            # Reset the weights
            weights = np.ones(nreplicas) / float(nreplicas)
            log_weights = np.log(weights)

            # Update with Langevin dynamics
            seed = advance_replicas(replica_unknowns, nn, nsteps=nsteps, seed=seed)
    print('100 % complete.')
    return nn, replica_unknowns, sample_sizes, resample_times


def smc_geometric_likelihood(inputs, target, nreplicas, layer_shapes, nstages, draw_param_scale, draw_error_scale,
                             ess_frac=0.7, nsteps=100, seed=0, loud=True):
    """
    Function to perform sequential Monte Carlo fitting of a neural net in which the data is geometrically
    annealed into the posterior.

    Parameters
    ----------
    inputs: array like
        The explanatory variables, where each variable is in a different column.
    target: array like
        The response variable.
    nreplicas: int
        The number of SMC particles.
    layer_shapes: tuple of ints
        The number of input layers, hidden layers, and output layers of the
    nstages: int
        The number of annealing steps to make.
    draw_param_scale: float
        The stanadard deviation of the priors over the NN weights and biases. The priors have a mean of zero.
    draw_error_scale: float
        The scale parameter of the gamma prior on the standard deviation of the likelihood.
    ess_frac: float between 0 and 1.
        The minimum allowed fractional effective sample size. A value below this triggers resampling of the replicas
        and Langevin dynamics.
    nsteps: int
        The number of Langevin dynamics steps taken on each replica after resampling.
    seed: int
        The random seed.
    lout: bool
        Whether to occasionally print out status of the sampler.

    Returns
    -------
    nn: GaussianDenseNN
        The Bayesian nueral network that has been powering the sampling.
    replica_unknowns: list of jax arrays
        The parameters of the SMC replicas/particules. Can be slotted into the nn.
    resample_times: list
        The times when resampling was triggered.
    """
    # Sanity checks:
    if layer_shapes[0] != inputs.shape[1]:
        raise Exception(
            'The number of input layers ({0}) should match the number of explanatory variables ({1})'.format(
                layer_shapes[0], inputs.shape[1]))
    if layer_shapes[-1] != target.shape[1]:
        raise Exception('The number of output layers ({0}) should match the number of response variables ({1})'.format(
            layer_shapes[-1], target.shape[1]))

    # Generate a lambda path from 0 to 1.
    lambdas = 100 ** (np.linspace(0, 1, nstages))
    lambdas -= lambdas[0]
    lambdas /= lambdas[-1]

    # Initialize the NN with the initial data packet
    inputs = jnp.array(inputs)
    target = jnp.array(target)

    nn = dense_nn.GaussianDenseNN(inputs, target, layer_shapes,
                                  prior_weight_scale=draw_param_scale, prior_error_scale=draw_error_scale, seed=seed)
    nn.reinit_integrator(dt=0.0001, gamma=100, mass=10)
    seed += 1

    # Generate the particles
    replica_unknowns, seed = generate_replicas(nn, nreplicas, seed=seed)

    # Initialize the weights
    log_weights = np.zeros(nreplicas)  # Assumes the replicas have been drawn from the prior
    sample_sizes = np.zeros(nstages)
    resample_times = []

    # Print the status every this percent:
    percent = 10
    completion_freq = int(nstages * percent / 100)
    k = 1
    for n in range(1, nstages):

        nn.lam = lambdas[n]
        nn.reinit_system()

        # Tricky to optimize because the sigma value will change from iteration to iteration.
        log_weights, weights = calc_weights_geometric(nn, lambdas[n] - lambdas[n - 1], replica_unknowns,
                                                                   log_weights)

        s = ess(weights)
        # The expected number of progenitors.
        ancestor_frac = np.sum(np.floor(weights * nreplicas) > 0) / nreplicas

        if loud:
            if n % completion_freq == 0:
                print('{0}% completed. ESS = {1:.1f}, Progenitors={2:.1f}'.format(k * percent, s, ancestor_frac))
                k += 1

        if s < ess_frac * nreplicas:
            if loud:
                print('  Resampled at stage {0} of {1}'.format(n, nstages))
            resample_times.append(n)

            # Resample with replacement
            replica_unknowns = resample_replicas(weights, replica_unknowns)

            # Reset the weights
            weights = np.ones(nreplicas) / float(nreplicas)
            log_weights = np.log(weights)

            # Update with Langevin dynamics
            seed = advance_replicas(replica_unknowns, nn, nsteps=nsteps, seed=seed)
    print('100 % complete.')
    return nn, replica_unknowns, resample_times

