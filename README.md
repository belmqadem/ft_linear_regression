# ft_linear_regression

An introduction to machine learning — implementing linear regression with
batch gradient descent **from scratch**.

This project trains a model to predict the price of a car based on its
mileage, using a dataset of (mileage, price) pairs.

---

## Table of contents

- [Overview](#overview)
- [The math](#the-math)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Bonus features](#bonus-features)
- [Useful learning links](#useful-learning-links)

---

## Overview

Two entry-point programs, backed by a shared `utils` package and a `bonus`
package:

| Program            | Purpose                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `train_model.py`   | Reads `data/data.csv`, runs gradient descent, saves the learned parameters to `model.json` |
| `predict_price.py` | Prompts for a mileage, loads `model.json`, prints the estimated price                      |

If `model.json` doesn't exist yet (model not trained), `predict_price.py`
falls back to θ0 = θ1 = 0, as required by the subject.

---

## The math

**Hypothesis** — the linear function used to predict price from mileage:

```
estimatePrice(mileage) = θ0 + (θ1 * mileage)
```

- `θ0` — intercept (bias)
- `θ1` — slope (weight on mileage)

**Cost function** — Mean Squared Error, measures how wrong the model is:

```
J(θ) = (1 / 2m) * Σ (estimatePrice(x[i]) - y[i])²
```

**Gradient descent update rules** — given by the subject:

```
tmpθ0 = learningRate * (1/m) * Σ (estimatePrice(x[i]) - y[i])
tmpθ1 = learningRate * (1/m) * Σ (estimatePrice(x[i]) - y[i]) * x[i]

θ0 := θ0 - tmpθ0
θ1 := θ1 - tmpθ1
```

Both `tmpθ0` and `tmpθ1` are computed using the **same, old** θ values, then
applied **simultaneously**.

**Feature normalization** — raw mileage (e.g. 240,000) and price (e.g. 3,650)
are on very different scales. Without normalization, gradient descent
diverges or converges far too slowly. Z-score normalization is applied:

```
x_normalized = (x - mean(x)) / std(x)
```

`mean` and `std` are saved alongside θ0/θ1 in `model.json`, since the
prediction program must apply the exact same transformation to new inputs.

---

## Project structure

```
ft_linear_regression/
├── train_model.py             # training entry point
├── predict_price.py           # prediction entry point
├── model.json                 # generated — θ0, θ1, mean, std
├── data/
│   └── data.csv                # training dataset (mileage, price)
├── utils/
│   └── helpers.py               # shared functions: I/O, math, hypothesis, cost
├── bonus/
│   ├── calculate_precision.py   # R² score
│   └── plot_data.py              # data + regression line + cost curve plot
├── training_result.png         # generated — bonus plot
└── .gitignore
```

### `utils/helpers.py`

Shared building blocks used by both entry points:

- `read_dataset` — load mileage/price pairs from CSV
- `mean`, `std`, `normalize` — feature scaling utilities
- `estimate_price` — the hypothesis `θ0 + θ1 * mileage`
- `compute_cost` — MSE cost function J(θ)
- `load_model` — load θ0, θ1, mu_km, sigma_km from `model.json` (or defaults to 0,0,0,1 if untrained)

### `bonus/`

- `calculate_precision.py` — computes R² to measure model accuracy
- `plot_data.py` — generates `training_result.png` (scatter + regression line + cost convergence curve)

---

## Setup

```bash
# clone and enter the project
git clone <your-repo-url>
cd ft_linear_regression

# (optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies (only needed for the bonus plot)
pip install matplotlib
```

No dependencies are required for the mandatory part — only the Python
standard library (`csv`, `json`, `math`).

---

## Usage

### 1. Train the model

```bash
python3 train_model.py
```

This will:

1. Load `data/data.csv`
2. Normalize the mileage column
3. Run batch gradient descent (default: α = 0.1, 1000 iterations)
4. Print θ0, θ1, final cost, and R² score
5. Save the model to `model.json`
6. Generate `training_result.png` (if matplotlib is installed)

### 2. Predict a price

```bash
python3 predict_price.py
```

```
Enter mileage (km): 150000
Estimated price for 150,000 km is 5,271.53 €
```

Run `predict_price.py` **before** training to confirm the θ0 = θ1 = 0
fallback works correctly (every prediction returns 0).

---

## How it works

```
┌─────────────────┐      ┌────────────────────────┐     ┌──────────────┐
│ data/data.csv    │ ──▶ │  train_model.py        │ ──▶ │ model.json   │
│ km, price        │     │  - normalize (helpers) │     │ θ0, θ1,      │
└─────────────────┘      │  - gradient descent    │     │ mu, sigma    │
                         │  - r² (bonus)          │     └──────────────┘
                         │  - plot (bonus)        │             │
                         └────────────────────────┘             ▼
                                                       ┌─────────────────────┐
                                                       │ predict_price.py    │
                                                       │  - load model       │
                                                       │  - normalize input  │
                                                       │  - estimate_price() │
                                                       └─────────────────────┘
```

Both entry points import their shared logic from `utils.helpers`, keeping
the hypothesis, cost function, and I/O in one place — no duplicated math.

---

## Bonus features

- ✅ **Plot data + regression line** — `bonus/plot_data.py` generates
  `training_result.png`, automatically skipped if `matplotlib` is missing
- ✅ **Cost convergence curve** — visualizes gradient descent converging
  over iterations, in the same plot
- ✅ **Precision metric (R²)** — `bonus/calculate_precision.py`, printed
  after training. Measures how much of the price variance the model
  explains (0 = no better than guessing the mean, 1 = perfect fit)

---

## Useful learning links

**Core concept — Stanford CS229 (Andrew Ng)**

- [Lecture 2: Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8) — the theoretical foundation for this entire project. Covers the hypothesis, cost function, and the simultaneous-update gradient descent rule.

**Gradient descent intuition**

- [3Blue1Brown — Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — visual intuition for what "descending" a cost function actually means.
- [StatQuest — Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — slower-paced, example-driven walkthrough.

**Cost functions and R²**

- [StatQuest — R-squared explained](https://www.youtube.com/watch?v=2AQKmw14mHM) — clear explanation of what R² actually measures.
