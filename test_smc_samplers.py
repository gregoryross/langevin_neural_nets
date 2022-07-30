import dense_nn
import numpy as np
import smc_samplers
import copy


def test_gen_data_splits():
    N = 300
    ncolumns = 3
    ndata = int(N / ncolumns)
    inputs = np.random.normal(loc=0.0, scale=1, size=N).reshape(ndata, ncolumns)
    target = np.random.normal(loc=0.0, scale=1, size=ndata).reshape(ndata, 1)

    start_points, end_points = smc_samplers.gen_data_splits(ndata, packet_size=9)

    assert len(start_points) == len(end_points)

    # Test the concatenation of the inputs
    data = []
    for s, e in zip(start_points, end_points):
        data.extend(target[s:e])
    assert np.all(np.array(data) == target)

    # Test the concatenation of the targets
    data = []
    for s, e in zip(start_points, end_points):
        data.extend(inputs[s:e])
    assert np.all(np.array(data) == inputs)

def test_calc_weights_geometric():
    """
    Make sure that the weights are being updated correctly for the geometric SMC method.
    """
    # Creata a quick NN:
    N = 200
    ncolumns = 2
    ndata = int(N / ncolumns)
    inputs = np.random.normal(loc=0.0, scale=1, size=N).reshape(ndata, ncolumns)
    target = np.random.normal(loc=0.0, scale=1, size=ndata).reshape(ndata, 1)
    nn = dense_nn.GaussianDenseNN(inputs, target, (ncolumns, 5, 1), prior_weight_scale=0.5, prior_error_scale=0.5)

    # Create some initial weights
    nreplicas = 3
    replica_unknowns = [nn.unknowns] * nreplicas
    initial_weights = np.repeat(1 / nreplicas, nreplicas)
    log_weights = np.log(initial_weights)
    initial_log_weights = log_weights.copy()

    lam1 = 0.1
    lam2 = 0.5
    dlam = lam2 - lam1

    # Calculate the new weights
    new_log_weights, weights = smc_samplers.calc_weights_geometric(nn, dlam, replica_unknowns, log_weights)

    # Calculate what the difference between the old and new weights should be:
    diff_log_weights = float(nn.batch_log_like(replica_unknowns[0], nn.inputs, nn.targets, lam=lam2)) - \
                       float(nn.batch_log_like(replica_unknowns[0], nn.inputs, nn.targets, lam=lam1))

    assert np.abs(new_log_weights[0] - initial_log_weights[0] - diff_log_weights) < 0.0001

def test_generate_replicas():
    """
    Make sure SMC particles/replicas gel with the NN.
    """
    # Creata a quick NN:
    N = 50
    ncolumns = 2
    ndata = int(N / ncolumns)
    inputs = np.random.normal(loc=0.0, scale=1, size=N).reshape(ndata, ncolumns)
    target = np.random.normal(loc=0.0, scale=1, size=ndata).reshape(ndata, 1)
    nn = dense_nn.GaussianDenseNN(inputs, target, (ncolumns, 5, 1), prior_weight_scale=0.5, prior_error_scale=0.5)

    nreplicas = 3
    seed = 0
    old_seed = copy.deepcopy(seed)
    replica_unknowns, seed = smc_samplers.generate_replicas(nn, nreplicas, seed)

    assert seed != old_seed

    energies = []
    for i in range(nreplicas):
        nn.unknowns = replica_unknowns[i]
        energies.append(nn.energy(nn.unknowns))
    energies = np.array(energies)

    assert np.all(np.isreal(energies))

    assert replica_unknowns[0].shape == nn.unknowns.shape
    assert np.all(replica_unknowns[0] != replica_unknowns[1])