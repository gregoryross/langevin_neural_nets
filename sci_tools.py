import itertools
import numpy as np


def poly(inputs, terms, coefficients):
    """
    Applies a polynomial to the inputs to return a 1D output. Function has the form
    Y = a + b*X_1 + c*X_2 + ...    d*X_1*X_2 + ... + e*X_1*X_2X_3 ...

    Parameters
    ----------
    terms: numpy.ndarray
        The input variables used in the polynomial. Eg. [(0,), (1,2), (1,2,3)] produces a function of the form
        Y = a + b*X[:,0] + c*X[:,1]*X[:,2] + d*X[:,1]*X[:,2]*X[:,3].
    coefficients: numpy.ndarray
        The coefficients of the variables including an additional offset term.

    Returns
    -------
    output: numpy.ndarray
        The 1D output of the polynomial.

    """
    output = np.zeros(inputs.shape[0])
    for i in range(len(terms)):
        output += coefficients[i] * inputs[:, terms[i]].cumprod(axis=1)[:, -1]
    output += coefficients[-1]
    return output


def gen_rand_poly(dim, degree, nterms):
    """
    Generate random factors and coefficients for a polynomial.

    TODO
    ----
    Does not generate self powers, e.g. x^2, x^3, etc, only multiplicative powers x_1*x_2, x_2*x_3 etc.

    Parameters
    ----------
    dim: int
        The dimension of the input.
    degree: int
        The degree of the polynomial.
    nterms: int
        The number of terms, excluding the offset in the polynomial.
    """
    variables = []
    # Get all singletons and cross terms, e.g. X1*X2, X1*X2*X3
    for i in range(1, degree + 1):
        variables.extend(list(itertools.combinations(range(dim), i)))
    # Get all self terms, e.g. X1^2, X1^3, etc
    for d in range(dim):
        for p in range(2, degree + 1):
            variables.append(tuple(d for i in range(p)))

    if nterms > len(variables):
        nterms = len(variables)

    indices = np.random.choice(range(len(variables)), nterms, replace=False)
    terms = [variables[indx] for indx in indices]
    coefficients = np.random.uniform(-10, 10, size=(nterms + 1))  # Generating an additional term for the offset

    def rand_poly(inputs):
        return poly(inputs, terms, coefficients)

    return rand_poly, (terms, coefficients)


def binary_search(binary_f, min_val=0, max_val=1, max_iter=20, initial_guess=None, precision_threshold=0):
    """
    binary_f is False from min_val up to unknown_val,
    then True from unknown_val up to max_val. Find unknown val.

    Written by Josh Fass
    """

    intervals = [(min_val, max_val)]
    mid_point = 0.5 * (min_val + max_val)
    if type(initial_guess) != type(None):
        mid_point = initial_guess

    for i in range(max_iter):
        if binary_f(mid_point):
            max_val = mid_point
        else:
            min_val = mid_point
        intervals.append((min_val, max_val))
        mid_point = (min_val + max_val) * 0.5
        if (max_val - min_val) <= precision_threshold:
            break
    return mid_point, intervals