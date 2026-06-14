from utils.helpers import load_model
from utils.helpers import estimate_price


def predict_price():
    """
    Program to predict the price of a car for a given mileage.
    1. Prompt the user for a mileage value
    2. Calculate the price using the linear regression model
    3. Print the predicted price
    """

    MODEL_PATH = "model.json"

    theta0, theta1, mu_km, sigma_km = load_model(MODEL_PATH)

    while True:
        try:
            raw = input("Enter mileage (km): ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return
        try:
            mileage = float(raw)
            if mileage < 0:
                print("  Mileage cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print(f"  '{raw}' is not a valid number. Please try again.")

    mileage_norm = (mileage - mu_km) / sigma_km

    price = estimate_price(theta0, theta1, mileage_norm)

    print(f"\nEstimated price for {mileage:,.0f} km is {price:,.2f} €")


if __name__ == "__main__":
    predict_price()
