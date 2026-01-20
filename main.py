import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils.validation
import tbats.abstract.Estimator
from tbats import TBATS
from sklearn.metrics import mean_squared_error
from math import sqrt
import json

_real_check_array = sklearn.utils.validation.check_array


def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _real_check_array(*args, **kwargs)


# Force overwrite inside tbats module
tbats.abstract.Estimator.check_array = _patched_check_array


# ---------------------------------------------------------
# 2. HELPER: Scale Bar Plotter (For Figure 3 replication)
# ---------------------------------------------------------
def add_scale_bar(ax, data_range, label_size=0.1):
    """
    Adds a grey bar on the right side of the plot to allow
    visual comparison of magnitudes between subplots.
    """
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    bar_height = data_range * 0.15

    mid_y = ymin + (yrange / 2)

    xmin, xmax = ax.get_xlim()
    x_pos = xmax + (xmax - xmin) * 0.01

    ax.plot(
        [x_pos, x_pos],
        [mid_y - bar_height / 2, mid_y + bar_height / 2],
        color="gray",
        linewidth=4,
        clip_on=False,
    )


# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":

    # --- A. LOAD DATA ---
    # url ='https://robjhyndman.com/data/gasoline.csv'
    url = "dataset.csv"
    try:
        df = pd.read_csv(url)
        y_all = df["values"].values[:745].astype(float)

        # --- B. SPLIT DATA ---
        n_test = 261
        n_train = len(y_all) - n_test

        y_train = y_all[:n_train]
        y_test = y_all[n_train:]

        print(f"Total Observations: {len(y_all)}")
        print(f"Training Set: {len(y_train)}")
        print(f"Test Set:     {len(y_test)}")

        # Create Dates
        start_date = pd.Timestamp("1991-02-02")
        dates_all = pd.date_range(start=start_date, periods=len(y_all), freq="W")
        dates_train = dates_all[:n_train]
        dates_test = dates_all[n_train:]

        print("\nFitting TBATS model (Training set)...")

        # --- C. FIT MODEL ---
        estimator = TBATS(
            seasonal_periods=[365.25 / 7],
            use_box_cox=True,
            use_trend=True,
            use_damped_trend=False,
            use_arma_errors=True,
            # use_parallel=False,  # Stability
        )

        model = estimator.fit(y_train)

        # --- D. EXTRACT PARAMETERS ---
        p = model.params
        print("\n" + "=" * 40)
        print("          TBATS COEFFICIENTS")
        print("=" * 40)
        print(f"w (Box-Cox Lambda): {p.box_cox_lambda:.6f}")
        print(f"a (Alpha - Level):  {p.alpha:.6f}")
        print(f"b (Beta - Trend):   {p.beta:.6f}")
        print(f"f (Phi - Damping):  {p.phi:.6f}")

        # ARMA Coefficients
        # print(f"ARMA (p,q):         ({p.p}, {p.q})")
        # if p.p > 0:
        #     print(f"AR Coefficients:    {model.params.ar_coefs}")
        # if p.q > 0:
        #     print(f"MA Coefficients:    {model.params.ma_coefs}")

        print(f"Gamma 1 (Seas 1):   {p.gamma_1()}")
        print(f"Gamma 2 (Seas 2):   {p.gamma_2()}")
        # json_string = json.dumps(p.__dict__, indent=4)
        # print(json_string)
        print(f"AR Coefs:           {model.params.ar_coefs}")
        print(f"Theta (MA):			{model.params.ma_coefs}")
        print("=" * 40)

        # --- E. FORECAST & RMSE ---
        print(f"\nForecasting {n_test} steps ahead...")
        y_fc, conf_int = model.forecast(steps=n_test, confidence_level=0.95)

        # Calculate RMSE
        rmse_val = sqrt(mean_squared_error(y_test, y_fc))
        print(f"Out-of-Sample RMSE: {rmse_val:.4f}")

        # --- F. PLOT 1: FORECAST VS ACTUAL (RMSE Visualization) ---
        plt.figure(figsize=(12, 6))
        plt.plot(
            dates_train[-100:],
            y_train[-100:],
            label="Training Data (Last 100)",
            color="black",
            alpha=0.5,
        )
        plt.plot(
            dates_test, y_test, label="Actual Test Data", color="green", linewidth=1.5
        )
        plt.plot(
            dates_test,
            y_fc,
            label=f"TBATS Forecast (RMSE={rmse_val:.2f})",
            color="red",
            linestyle="--",
        )

        # Shade the error area to visualize RMSE source
        plt.fill_between(
            dates_test, y_test, y_fc, color="red", alpha=0.1, label="Error Magnitude"
        )

        plt.title(f"Out-of-Sample Forecast (p={n_test})")
        plt.ylabel("Barrels per day")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        y_fc, conf_int = model.forecast(steps=n_test, confidence_level=0.95)
        squared_errors = (y_test - y_fc) ** 2
        cumulative_mse = np.cumsum(squared_errors) / np.arange(
            1, len(squared_errors) + 1
        )
        cumulative_rmse = np.sqrt(cumulative_mse)

        weeks = np.arange(1, n_test + 1)

        plt.figure(figsize=(10, 6))

        # Plot the curve
        plt.plot(
            weeks, cumulative_rmse, color="purple", linewidth=2, label="Cumulative RMSE"
        )

        # Add a horizontal line for the final overall RMSE value
        final_rmse = cumulative_rmse[-1]
        plt.axhline(
            final_rmse,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Final RMSE ({final_rmse:.2f})",
        )

        plt.title("Out-of-Sample Performance: Cumulative RMSE over Horizon")
        plt.ylabel("RMSE (Root Mean Square Error)")
        plt.xlabel("Forecast Horizon (Weeks)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        fitted_values = model.y_hat
        residuals = model.resid

        # Prepare subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Calculate global range for scale comparison
        global_range = np.max(y_train) - np.min(y_train)

        # 1. Observed
        axes[0].plot(dates_train, y_train, color="black", linewidth=1)
        axes[0].set_ylabel("Observed")
        axes[0].set_title("Figure 3 Replication: TBATS Decomposition")
        add_scale_bar(axes[0], global_range)

        # 2. Fitted (Level + Season + Trend)
        axes[1].plot(dates_train, fitted_values, color="blue", linewidth=1)
        axes[1].set_ylabel("Fitted (Level+Season)")
        add_scale_bar(axes[1], global_range)

        # 3. Residuals
        axes[2].plot(dates_train, residuals, color="gray", linewidth=1)
        axes[2].axhline(0, color="black", linestyle="--", linewidth=0.5)
        axes[2].set_ylabel("Residuals")
        add_scale_bar(axes[2], global_range)

        plt.xlabel("Year")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        import traceback

        traceback.print_exc()
