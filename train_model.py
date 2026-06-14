import json
from utils.helpers import read_dataset
from utils.helpers import mean
from utils.helpers import std
from utils.helpers import normalize
from utils.helpers import estimate_price
from utils.helpers import compute_cost
from bonus.calculate_precision import r_squared
from bonus.plot_data import plot_results


def gradient_descent(
    x: list[float],
    y: list[float],
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> tuple[float, float, list[float]]:
    """
    Update rules:
      tmpθ0 = learningRate * (1/m) * Σ(estimatePrice(x[i]) - y[i])
      tmpθ1 = learningRate * (1/m) * Σ((estimatePrice(x[i]) - y[i]) * x[i])
      θ0 := θ0 - tmpθ0
      θ1 := θ1 - tmpθ1
    """

    theta0, theta1 = 0.0, 0.0
    m = len(x)
    cost_history = []

    for _ in range(iterations):
        tmp_theta0 = (
            learning_rate
            * (1 / m)
            * sum(estimate_price(theta0, theta1, x[i]) - y[i] for i in range(m))
        )
        tmp_theta1 = (
            learning_rate
            * (1 / m)
            * sum(
                (estimate_price(theta0, theta1, x[i]) - y[i]) * x[i] for i in range(m)
            )
        )

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        cost_history.append(compute_cost(theta0, theta1, x, y))

    return theta0, theta1, cost_history


def train_model():
    """
    Program to train a linear regression model to predict the price of a car based on its mileage.
    1. Read the dataset from a CSV file
    2. Perform a linear regression to find the optimal values of θ0 and θ1
    3. Save the trained model parameters (θ0 and θ1) for use in the prediction program
    """

    DATASET_PATH = "data/data.csv"
    MODEL_PATH = "model.json"
    LEARNING_RATE = 0.1
    ITERATIONS = 1000

    print("Loading dataset...")
    mileages, prices = read_dataset(DATASET_PATH)
    m = len(mileages)
    print(f"  {m} training examples loaded.")

    mu_km = mean(mileages)
    sigma_km = std(mileages)
    x_norm = normalize(mileages, mu_km, sigma_km)
    print(f"  Normalization — mean: {mu_km:.1f} km, std: {sigma_km:.1f} km")

    print(f"Running gradient descent (α={LEARNING_RATE}, {ITERATIONS} iterations)...")
    theta0, theta1, cost_history = gradient_descent(
        x_norm, prices, LEARNING_RATE, ITERATIONS
    )

    print(f"  θ0 = {theta0:.6f}")
    print(f"  θ1 = {theta1:.6f}")
    print(f"  Final cost J(θ) = {cost_history[-1]:.4f}")

    r2 = r_squared(theta0, theta1, x_norm, prices)
    print(f"  R² score = {r2:.4f}  ({r2*100:.1f}% of variance explained)")

    # save model
    model = {
        "theta0": theta0,
        "theta1": theta1,
        "mu_km": mu_km,
        "sigma_km": sigma_km,
    }
    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  Model saved → {MODEL_PATH}")

    # plot (bonus)
    plot_results(mileages, prices, theta0, theta1, mu_km, sigma_km, cost_history)


if __name__ == "__main__":
    train_model()
