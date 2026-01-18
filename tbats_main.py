import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbats import TBATS
import sklearn.utils.validation

# --- 1. MONKEY PATCH FOR SCIKIT-LEARN ERROR ---
# This fixes the "force_all_finite" error from your previous attempt
_original_check_array = sklearn.utils.validation.check_array


def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _original_check_array(*args, **kwargs)


sklearn.utils.validation.check_array = _patched_check_array
# ----------------------------------------------

# --- 2. MAIN EXECUTION BLOCK ---
# Everything that runs logic must be inside this if-statement
if __name__ == "__main__":

    # LOAD DATA
    url = "dataset.csv"
    try:
        df = pd.read_csv(url)
        y = df["values"].values.astype(float)
        y = pd.Series(y).interpolate().fillna(method="bfill").values
        y = np.array(y)
        print(y.shape)

        # Generate dates
        start_date = pd.Timestamp("1991-02-02")
        dates = pd.date_range(start=start_date, periods=len(y), freq="W")

        print("Data loaded. Fitting TBATS model...")
        print("(This will take time, but should not crash now...)")

        # FIT TBATS
        # We define the seasonality here
        estimator = TBATS(
            use_box_cox=True,
            use_trend=True,
            use_damped_trend=True,
            use_arma_errors=True,
            seasonal_periods=[52],
        )

        # This line triggers multiprocessing, which is safe now due to the if-block
        gastbats = estimator.fit(y)

        print("\nModel Fitted Successfully!")
        print(f"Box-Cox Lambda: {gastbats.params.box_cox_lambda:.4f}")

        # FORECAST
        h = 104
        y_forecast, conf_int = gastbats.forecast(steps=h, confidence_level=0.95)

        # PLOT
        future_dates = pd.date_range(start=dates[-1], periods=h + 1, freq="W")[1:]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, y, label="Observed", color="black", linewidth=1)
        plt.plot(future_dates, y_forecast, label="Forecast", color="blue")
        plt.fill_between(
            future_dates,
            conf_int["lower_bound"],
            conf_int["upper_bound"],
            color="blue",
            alpha=0.2,
            label="95% CI",
        )

        plt.ylabel("Thousands of barrels per day")
        plt.title("TBATS Forecast (US Gasoline)")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.show()

    except RuntimeError:
        print(
            "A runtime error occurred. If this is a multiprocessing error, ensure the code is wrapped in 'if __name__ == \"__main__\":'"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
