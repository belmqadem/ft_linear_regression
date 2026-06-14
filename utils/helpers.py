import csv
import json
import math
import os


def load_model(path: str) -> tuple[float, float, float, float]:
    if not os.path.exists(path):
        print(f"  [info] '{path}' not found -—> model not trained yet.")
        return 0.0, 0.0, 0.0, 1.0

    with open(path) as f:
        data = json.load(f)

    theta0 = data["theta0"]
    theta1 = data["theta1"]
    mu_km = data["mu_km"]
    sigma_km = data["sigma_km"]
    return theta0, theta1, mu_km, sigma_km


def read_dataset(path: str) -> tuple[list[float], list[float]]:
    mileages, prices = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))
    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def normalize(values: list[float], mu: float, sigma: float) -> list[float]:
    return [(v - mu) / sigma for v in values]


def estimate_price(theta0: float, theta1: float, mileage: float) -> float:
    """hypothesis: estimatePrice(mileage) = θ0 + (θ1 * mileage)"""
    return theta0 + (theta1 * mileage)


def compute_cost(theta0: float, theta1: float, x: list[float], y: list[float]) -> float:
    """MSE cost function: J = (1/2m) * Σ(h(x) - y)²"""
    m = len(x)
    total = sum((estimate_price(theta0, theta1, x[i]) - y[i]) ** 2 for i in range(m))
    return total / (2 * m)
