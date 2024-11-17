import numpy as np


def polynomial_1d(coefficients, inputs):
    """
    Evaluate a polynomial with given coefficients on an array of inputs.

    Args:
        coefficients (array-like): Coefficients of the polynomial, starting with the
            coefficient for the highest degree term and ending with the coefficient for
            the constant term.
        inputs (array-like): Inputs at which to evaluate the polynomial.

    Returns:
        outputs (ndarray): Outputs of the polynomial at the given inputs.
    """
    # Create a polynomial object from the coefficients
    polynomial = np.poly1d(coefficients)

    # Evaluate the polynomial at the given inputs
    outputs = polynomial(inputs)

    return outputs
