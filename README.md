# TBATS Model Replication: US Gasoline Data

**Replication of De Livera, Hyndman, & Snyder (2011)**

## 📌 Project Overview

This project replicates the analysis performed in Case Study 1 of the paper _"Forecasting time series with complex seasonal patterns using exponential smoothing"_ (De Livera et al., 2011).

The specific focus is on the **US Finished Motor Gasoline Product Supplied** dataset, utilizing the **TBATS** (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) model to forecast weekly gasoline demand.

The goal is to reproduce the results obtained in R (using the `forecast` package) using the **Python** ecosystem (using the `tbats` library).

## 📄 Key Objectives

1.  **Replicate Figure 1(b):** Visualize the raw time series (Feb 1991 – July 2005).
2.  **Model Fitting:** Fit a TBATS model on the training set (484 observations) with specific constraints to match the paper ($\lambda \approx 1$, MA(1) errors).
3.  **Forecasting:** Predict the next 261 weeks (Test set).
4.  **Evaluation:** Calculate RMSE and visualize performance.
5.  **Replicate Figure 3:** Visualize the decomposition of the time series (Observed, Level/Trend, Seasonality, Residuals).

## 🛠️ Technical Implementation & Challenges

### 1. Library Compatibility (Monkey Patch)

The Python `tbats` library has not been updated to support `scikit-learn` versions > 1.5, causing a `TypeError: check_array() got an unexpected keyword argument 'force_all_finite'`.

- **Solution:** The script includes a custom "Monkey Patch" that intercepts calls to `check_array` and translates the arguments to be compatible with modern `scikit-learn`.

### 2. Numerical Optimization (R vs. Python)

The original paper uses R's internal optimizer (C/Fortran based). Python uses `scipy.optimize`.

- **Observation:** Unconstrained, Python converges to a Box-Cox $\lambda \approx 0.66$.
- **Constraint:** To replicate the paper (which found $\lambda = 0.9922$), we explicitly force `use_box_cox=False` (Linearity) in the Python script to align the model structure.

## 📦 Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Simply run the main script. Since the dataset from Rob Hyndman's repository is included in the repository and generate the analysis.

```bash
python main.py
```

## 📊 Outputs

The script generates the following insights in the console and via Matplotlib windows:

1.  **Model Coefficients:**
    - Prints Box-Cox $\lambda$, Alpha, Beta, and ARMA(p,q) coefficients.
    - _Expected:_ $\lambda=1.0$, MA(1) error structure.
2.  **Forecast Plot:**
    - Visualizes Training data vs. Actual Test data vs. TBATS Forecast.
    - Displays the overall RMSE.
3.  **RMSE Evolution Plot:**
    - Shows how the Cumulative Root Mean Square Error changes over the forecast horizon.
4.  **Decomposition Plot (Figure 3):**
    - Vertical decomposition of Observed, Fitted, and Residual values.
    - Includes **Relative Scale Bars** (grey bars on the right) to compare the magnitude of components.

## 📝 Comparison of Results

| Parameter               | Original Paper (R) | Python Implementation |
| :---------------------- | :----------------- | :-------------------- |
| **Box-Cox ($\lambda$)** | 0.9922             | 1.0000 (Forced)       |
| **Trend**               | Additive           | Additive              |
| **ARMA Errors**         | MA(1)              | MA(1)                 |
| **Alpha ($\alpha$)**    | 0.0478             | ~0.0939               |
| **Theta ($\theta_1$)**  | -0.2124            | ~ -0.2581             |

_Note: While the model structure (MA terms, Trend) matches the paper, the Python implementation yields a higher RMSE due to differences in the non-convex optimization engines between R and Python._

## 📚 References

- **Paper:** De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). _Forecasting time series with complex seasonal patterns using exponential smoothing_. Journal of the American Statistical Association.
- **Dataset:** US Energy Information Administration.
