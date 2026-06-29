"""
Part 1 - Task 3: Time Series Generation, Decomposition & Forecasting

Dataset: synthetic time series of 365 days (train) + 60 days (test).
  Components: linear upward trend + weekly (7-day) seasonality + Gaussian noise.

Sub-tasks:
  1. Data generation & visualisation
  2. Decomposition (additive + multiplicative, period=7)
  3. Forecasting: 7-day Moving Average, Simple Exponential Smoothing, ARIMA(1,1,1)
  4. Model evaluation: MAE, MSE, MAPE
  5. Interpretation (covered in the responses document)
All figures saved next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    # --------------------------------------------------- 1. data generation
    n_total = 365 + 60
    t = np.arange(n_total)
    trend = 50 + 0.10 * t                       # linear upward trend
    seasonality = 10 * np.sin(2 * np.pi * t / 7)  # weekly sine wave
    noise = np.random.normal(0, 3, n_total)       # mean 0, std 3
    series_values = trend + seasonality + noise

    dates = pd.date_range("2024-01-01", periods=n_total, freq="D")
    series = pd.Series(series_values, index=dates, name="value")

    # always-positive copy for the multiplicative model
    series_pos = series.clip(lower=1.0)

    train = series.iloc[:365]
    test = series.iloc[365:]
    print(f"Total points: {n_total} | train: {len(train)} | test: {len(test)}")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(series.index, series.values, color="steelblue", lw=.9)
    ax.axvline(train.index[-1], color="red", ls="--", label="train/test split")
    ax.set_title("Synthetic Daily Time Series (trend + weekly seasonality + noise)")
    ax.set_xlabel("Date"); ax.set_ylabel("Value"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_timeseries.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------- 2. decomposition
    add = seasonal_decompose(series, model="additive", period=7)
    mul = seasonal_decompose(series_pos, model="multiplicative", period=7)

    for name, dec in [("additive", add), ("multiplicative", mul)]:
        fig = dec.plot()
        fig.set_size_inches(11, 8)
        fig.suptitle(f"Seasonal decomposition ({name}, period=7)", y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"fig2_decompose_{name}.png"), dpi=120)
        plt.close(fig)

    print("Additive residual mean   :", np.nanmean(add.resid))
    print("Multiplicative resid mean:", np.nanmean(mul.resid))

    # ------------------------------------------------- 3. forecasting (h=60)
    h = len(test)

    # (a) 7-day Moving Average: flat forecast = mean of last 7 training points
    ma_value = train.iloc[-7:].mean()
    ma_forecast = pd.Series(ma_value, index=test.index)

    # (b) Simple Exponential Smoothing
    ses_fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    ses_forecast = ses_fit.forecast(h)

    # (c) ARIMA(1,1,1)
    arima_fit = ARIMA(train, order=(1, 1, 1)).fit()
    arima_forecast = arima_fit.forecast(h)

    forecasts = {"Moving Average (7d)": ma_forecast,
                 "SES": ses_forecast,
                 "ARIMA(1,1,1)": arima_forecast}

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(train.index[-90:], train.iloc[-90:], color="gray", label="train (last 90d)")
    ax.plot(test.index, test.values, color="black", lw=1.8, label="actual (test)")
    for label, fc in forecasts.items():
        ax.plot(test.index, fc.values, ls="--", label=label)
    ax.axvline(train.index[-1], color="red", ls=":", alpha=.6)
    ax.set_title("60-day Forecast: Actual vs Predicted")
    ax.set_xlabel("Date"); ax.set_ylabel("Value"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_forecasts.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------- 4. evaluation
    rows = []
    for label, fc in forecasts.items():
        rows.append({
            "model": label,
            "MAE": mean_absolute_error(test, fc),
            "MSE": mean_squared_error(test, fc),
            "MAPE(%)": mape(test, fc),
        })
    metrics = pd.DataFrame(rows).set_index("model").round(3)
    print("\nForecast accuracy on the 60-day test set:\n", metrics)
    best = metrics["MAE"].idxmin()
    print(f"\nBest model by MAE: {best}")

    metrics.to_csv(os.path.join(HERE, "forecast_metrics.csv"))
    print("\nSaved figures + forecast_metrics.csv")


if __name__ == "__main__":
    main()
