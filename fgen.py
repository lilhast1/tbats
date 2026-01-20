import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import ssl
import certifi
import urllib.request

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
url = "dataset.csv"
# Read the single column of data
df = pd.read_csv(url)
print("kolone", df.columns)
y = df["values"].values
n_obs = len(y)

# Set up the time index roughly matching R's start=1991+31/365.25
# This is approximately Feb 2, 1991.
start_date = pd.Timestamp("1991-02-02")
dates = pd.date_range(start=start_date, periods=n_obs, freq="W")

# ---------------------------------------------------------
# 2. DEFINE FOURIER GENERATOR
# ---------------------------------------------------------
# The seasonal period in the R code is 365.25 / 7
period = 365.25 / 7


def fourier_terms(t_indices, k, p):
    """
    Generates Fourier terms (sine and cosine pairs).
    t_indices: integer array of time steps (1, 2, 3...)
    k: number of harmonic pairs
    p: period length
    """
    X = []
    for i in range(1, k + 1):
        # Sine term
        X.append(np.sin(2 * np.pi * i * t_indices / p))
        # Cosine term
        X.append(np.cos(2 * np.pi * i * t_indices / p))
    # Stack columns: [sin1, cos1, sin2, cos2, ...]
    return np.column_stack(X)


# Time indices for the existing data (1 to N)
t_idx = np.arange(1, n_obs + 1)

# ---------------------------------------------------------
# 3. OPTIMIZATION LOOP (Find Best K)
# ---------------------------------------------------------
print("Starting search for optimal harmonics (K)...")

best_aicc = float("inf")
best_k = 0
best_model = None

# Loop from 1 to 25 as per your R code
for k in range(1, 26):
    # Generate regressors for current k
    X_k = fourier_terms(t_idx, k, period)

    # Fit auto_arima
    # seasonal=False because seasonality is handled by X_k (Fourier terms)
    # stepwise=True mimics R's default behavior
    model = pm.auto_arima(
        y,
        X=X_k,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )

    current_aicc = model.aicc()

    print(f"K={k}: AICc={current_aicc:.2f}")

    # Check for improvement
    if current_aicc < best_aicc:
        best_aicc = current_aicc
        best_model = model
        best_k = k
    else:
        print(f"AICc did not improve at K={k}. Stopping early.")
        break

print(f"\nBest Model Found: K={best_k} with AICc={best_aicc:.2f}")

# ---------------------------------------------------------
# 4. FORECASTING
# ---------------------------------------------------------
h = 104  # Forecast horizon (approx 2 years)

# Generate Future Time Indices and Fourier Terms
t_future = np.arange(n_obs + 1, n_obs + 1 + h)

# Note: In your R snippet, you hardcoded K=12 in the forecast line:
# fc <- forecast(bestfit, xreg=fourier(gas, K=12, h=104))
# However, usually, one uses the 'best_k' found in the loop.
# We will use 'best_k' to ensure the matrix shapes match the trained model.
X_future = fourier_terms(t_future, best_k, period)

# Predict
forecast_values, conf_int = best_model.predict(
    n_periods=h,
    X=X_future,
    return_conf_int=True,
    alpha=0.05,  # 95% Confidence Intervals
)

# ---------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------
# Create future dates for plotting
future_dates = pd.date_range(start=dates[-1], periods=h + 1, freq="W")[1:]

plt.figure(figsize=(12, 6))

# Plot historical data (zooming in on later years for clarity, similar to R plots)
plt.plot(dates, y, label="History", color="black", linewidth=1)

# Plot Forecast
plt.plot(future_dates, forecast_values, label="Forecast", color="blue")

# Plot Confidence Intervals
plt.fill_between(
    future_dates,
    conf_int[:, 0],
    conf_int[:, 1],
    color="blue",
    alpha=0.2,
    label="95% CI",
)

plt.title(f"Dynamic Harmonic Regression Forecast (K={best_k})")
plt.ylabel("US Gasoline Supplied")
plt.xlabel("Year")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
