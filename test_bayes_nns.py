import jax
import jax.numpy as jnp
import jaxlib
import dense_nn
import sci_tools
import numpy as np

"""
A set of unit tests.
"""

def gen_quick_model():
    # Generate some random input and output data
    N = 1000
    ncolumns = 2
    nrows = int(N / ncolumns)
    inputs = np.random.normal(loc=0.0, scale=1, size=N).reshape(nrows, ncolumns)
    rand_poly, (terms, coefficients) = sci_tools.gen_rand_poly(ncolumns, degree=3, nterms=3)
    target = rand_poly(inputs)


    # Instantiate a Baysian NN model
    inputs = jnp.array(inputs)
    target = jnp.array(target).reshape((nrows,1))
    layer_shapes = (inputs.shape[1], 50, 50, target.shape[1])
    nn = dense_nn.GaussianDenseNN(inputs, target, layer_shapes, prior_weight_scale=1, prior_error_scale=1)
    return nn


def test_nn_types():
    """
    Are the types produced by the NN sensical?
    """

    nn = gen_quick_model()
    test_energy = nn.energy(nn.unknowns)
    assert type(test_energy) == jaxlib.xla_extension.DeviceArray
    assert test_energy.size == 1

    test_grad = nn.gradient(nn.unknowns)
    assert type(test_grad) == jaxlib.xla_extension.DeviceArray
    assert test_grad.shape == nn.unknowns.shape


def test_rand_poly():
    pass


def test_nn_init():
    """
    Make sure the NN can still be initialized.
    """
    nn = gen_quick_model()
    assert nn is not None

def test_nn_energies():
    """
    Perform self consistency tests of a GaussianDenseNN object.
    Parameters
    ----------
    guass_nn: GaussianDenseNN object
    """
    # Create a test NN with random input data:
    gauss_nn = gen_quick_model()
    try:
        test_energy = gauss_nn.energy(gauss_nn.unknowns)
        assert type(test_energy) == jax.interpreters.xla.DeviceArray
        assert test_energy.size == 1
    except:
        print('The energy of the NN is not a single value in the correct format.')
    try:
        test_grad = gauss_nn.gradient(gauss_nn.unknowns)
        assert type(test_grad) == jax.interpreters.xla.DeviceArray
        assert test_grad.shape == gauss_nn.unknowns.shape
    except:
        print('The gradient of the NN does is not of the right type or shape.')
        # Make sure that the log likelihood function supports new data
    try:
        val = gauss_nn.batch_log_like(gauss_nn.unknowns, gauss_nn.inputs[[-1]], gauss_nn.targets[[-1]])
        assert val.size == 1
        assert val.dtype.name == 'float32'

        val = gauss_nn.batch_log_like(gauss_nn.unknowns, gauss_nn.inputs[1:3, :], gauss_nn.targets[1:3, :])
        assert val.size == 1
        assert val.dtype.name == 'float32'
    except:
        print('Unable to append new data to the likelihood function.')


def test_initialize_nn():
    """
    The only thing that matters is that it creates an array that is compatible with predict.
    """
    output_shape = 1
    input_shape = 2
    N = 100
    nrows = int(N / input_shape)
    inputs = np.random.normal(loc=0.0, scale=1, size=N).reshape(nrows, input_shape)

    layer_shapes = (input_shape, 10, 10, output_shape)
    params = dense_nn.init_nn_params(layer_shapes, 1.0, 0)
    output = dense_nn.nn_predict(params, inputs, nonlinearity=np.tanh)

    assert output.shape == (nrows, output_shape)

if __name__ == "__main__":

    test_nn_init()
    test_nn_energies()
    test_initialize_nn()