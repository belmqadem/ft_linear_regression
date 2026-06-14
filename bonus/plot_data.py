def plot_results(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    mu_km: float,
    sigma_km: float,
    cost_history: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed — skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ft_linear_regression", fontsize=14, fontweight="bold")

    # left: scatter + regression line
    ax1.scatter(mileages, prices, color="#1D9E75", label="Training data", zorder=3)

    x_line = [min(mileages), max(mileages)]
    y_line = [theta0 + theta1 * ((x - mu_km) / sigma_km) for x in x_line]
    ax1.plot(x_line, y_line, color="#7F77DD", linewidth=2, label="Regression line")

    ax1.set_xlabel("Mileage (km)")
    ax1.set_ylabel("Price (€)")
    ax1.set_title("Data distribution + model fit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # right: cost curve
    ax2.plot(cost_history, color="#EF9F27", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost J(θ)")
    ax2.set_title("Gradient descent convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_result.png", dpi=120, bbox_inches="tight")
    print("  [plot] saved → training_result.png")
    plt.close(fig)
