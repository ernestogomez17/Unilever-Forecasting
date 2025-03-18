# %% add WMAPE and R-squared
from typing import Optional, Union

import numpy as np

def _metric_protections(
    y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray]
) -> None:
    assert (weights is None) or (np.sum(weights) > 0), "Sum of weights cannot be 0"
    assert (weights is None) or (
        weights.shape == y.shape
    ), f"Wrong weight dimension weights.shape {weights.shape}, y.shape {y.shape}"


    
def wmape(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Weighted Mean Absolute Percentage Error (WMAPE)

    Calculates the Weighted Mean Absolute Percentage Error between `y` and `y_hat`.
    WMAPE measures the accuracy of a forecast as a percentage and is
    particularly useful in situations where the target values are on a large scale.

    $$ \mathrm{WMAPE}(\\mathbf{y}, \\mathbf{\hat{y}}) = \\frac{\\sum_{\\tau} |y_{\\tau} - \hat{y}_{\\tau}|}{\\sum_{\\tau} |y_{\\tau}|} $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `weights`: Optional, numpy array of weights.<br>
    `axis`: Optional, Axis to apply the metric.

    **Returns:**<br>
    `wmape`: numpy array or float, Weighted Mean Absolute Percentage Error.
    """
    _metric_protections(y, y_hat, weights)
    
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    if weights is not None:
        wmape = np.sum(weights * delta_y) / np.sum(weights * scale)
    else:
        wmape = np.sum(delta_y, axis=axis) / np.sum(scale, axis=axis)

    return wmape

# %% add R-squared (Coefficient of Determination)

def r_squared(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """R-squared (Coefficient of Determination)

    Calculates the R-squared value between `y` and `y_hat`.
    R-squared measures the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s).

    $$ R^2 = 1 - \\frac{\\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^2}{\\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \\bar{y})^2} $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `weights`: Optional, numpy array of weights.<br>
    `axis`: Optional, Axis to apply the metric.

    **Returns:**<br>
    `r_squared`: numpy array or float, Coefficient of determination.
    """
    _metric_protections(y, y_hat, weights)

    ss_res = np.sum((y - y_hat) ** 2, axis=axis)
    ss_tot = np.sum((y - np.mean(y, axis=axis)) ** 2, axis=axis)
    
    # Use the specialized _divide_no_nan_r2 function
    r2 = 1 - _divide_no_nan_r2(ss_res, ss_tot)

    return r2


def _divide_no_nan_r2(a, b):
    """Auxiliary function to handle divide by 0 in R-squared."""
    div = a / b
    # Check if div is an array or a scalar
    if isinstance(div, np.ndarray):
        div[div != div] = 0.0  # Handle NaN values in arrays
        div[div == float("inf")] = 0.0  # Handle infinite values in arrays
    elif np.isnan(div) or np.isinf(div):
        div = 0.0  # Handle scalar NaN or inf
    return div