from utils.helpers import mean
from utils.helpers import estimate_price


def r_squared(
    theta0: float,
    theta1: float,
    x_norm: list[float],
    y: list[float],
) -> float:
    """
    R² = 1 - SS_res / SS_tot where:
        - SS_res = Σ(y[i] - estimatePrice(x[i]))²
        - SS_tot = Σ(y[i] - y_mean)²
    R² = 1.0  → perfect fit
    R² = 0.0  → model is as good as predicting the mean
    R² < 0    → model is worse than the mean
    """

    y_mean = mean(y)
    ss_res = sum(
        (y[i] - estimate_price(theta0, theta1, x_norm[i])) ** 2 for i in range(len(y))
    )
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
    return 1 - (ss_res / ss_tot)
