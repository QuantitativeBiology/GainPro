import numpy as np
import numpy.testing as npt

from GenerativeProteomics.utils.metrics.metrics import rmse

def test_rmse():
    x_true = np.array([1.0, 2.0, np.nan])
    x_pred = np.array([0.9, 1.9, 1.5])
    mask = ~np.isnan(x_true)

    expected_rmse = np.sqrt(((0.1**2 + 0.1**2)) / 2)

    test_rmse = rmse(x_true=x_true, x_pred=x_pred, mask=mask)

    npt.assert_allclose(
        test_rmse,
        expected_rmse,
        rtol=1e-7,
        err_msg=(
            "RMSE computed over masked entries only. "
            f"Expected {expected_rmse}, got {test_rmse}. "
            "Check masking logic and aggregation."
        ),
    )
