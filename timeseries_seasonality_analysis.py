import os
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gmean, mstats
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SeasonalityMetrics:
    product_id: str
    product_name: str
    n_observations: int
    detected_period: Optional[int]
    candidate_periods: str
    seasonality_strength: Optional[float]
    trend_strength: Optional[float]
    seasonal_amplitude_abs: Optional[float]
    seasonal_amplitude_rel: Optional[float]
    seasonal_cv: Optional[float]
    seasonal_peak_month: Optional[str]
    seasonal_trough_month: Optional[str]
    yoy_growth_rate: Optional[float]
    cagr_12m: Optional[float]
    trend_slope_per_period: Optional[float]
    volatility_cv: Optional[float]
    concentration_gini: Optional[float]
    acf_lag1: Optional[float]
    acf_lag_detected: Optional[float]


def _safe_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x.fillna(0.0)


def _infer_period_via_acf(y: np.ndarray, min_period: int, max_period: int) -> Tuple[Optional[int], List[int]]:
    if len(y) < min_period * 2:
        return None, []

    # Compute ACF up to max_lag
    max_lag = min(max_period, max(2, len(y) // 2))
    acf_values = acf(y, nlags=max_lag, fft=True)
    # Skip lag 0
    lags = np.arange(len(acf_values))
    acf_values[0] = 0.0

    # Find local maxima peaks within the search window
    peaks, _ = find_peaks(acf_values, distance=2)
    candidate_mask = (lags >= min_period) & (lags <= max_period)
    candidate_peaks = [int(l) for l in peaks if candidate_mask[l]]

    # Rank peaks by acf value descending
    candidate_peaks_sorted = sorted(candidate_peaks, key=lambda l: acf_values[l], reverse=True)

    detected = candidate_peaks_sorted[0] if candidate_peaks_sorted else None
    return detected, candidate_peaks_sorted[:5]


def _strength_of_component(component: np.ndarray, remainder: np.ndarray) -> Optional[float]:
    if component is None or remainder is None:
        return None
    var_remainder = np.nanvar(remainder)
    var_total = np.nanvar(component + remainder)
    if var_total <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - var_remainder / var_total))


def _compute_gini(x: np.ndarray) -> float:
    # Gini coefficient of non-negative series
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    if np.allclose(x.sum(), 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(max(0.0, min(1.0, gini)))


def _theil_sen_slope(y: np.ndarray) -> float:
    # Robust trend slope per period (Theil–Sen estimator)
    n = len(y)
    if n < 3:
        return np.nan
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            denom = (j - i)
            if denom != 0:
                slopes.append((y[j] - y[i]) / denom)
    if not slopes:
        return np.nan
    return float(np.median(slopes))


def _month_name(idx: pd.Period) -> str:
    try:
        return idx.strftime("%b %Y")
    except Exception:
        return str(idx)


def prepare_dataframe(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    grouped = (
        df.groupby(["date", "sku", "mku"]).agg(
            {
                "mku_name": "first",
                **{col: "sum" for col in value_cols},
            }
        )
    ).reset_index()
    return grouped


def _seasonal_indices(series: pd.Series, period: int) -> Tuple[np.ndarray, np.ndarray]:
    # Compute multiplicative seasonal indices normalized to mean=1
    if period is None or period < 2:
        return np.array([]), np.array([])
    values = series.values.astype(float)
    # Add small epsilon to avoid divide by zero
    eps = 1e-12
    # STL with robust option to isolate seasonality
    stl = STL(values, period=period, robust=True)
    res = stl.fit()
    seasonal = res.seasonal
    # Convert additive seasonal to multiplicative indices when possible
    # Use ratio to seasonal-adjusted mean
    deseasonal = values - seasonal
    base_level = np.nanmean(deseasonal)
    if abs(base_level) < eps:
        base_level = np.nanmean(values) + eps
    seasonal_index = (values / (deseasonal + eps))
    # Normalize to mean 1 per cycle position
    idx_pos = np.arange(len(values)) % period
    pos_means = np.array([np.nanmean(seasonal_index[idx_pos == k]) for k in range(period)])
    pos_means[pos_means == 0] = 1.0
    seasonal_index_norm = seasonal_index / pos_means[idx_pos]
    # Aggregate stability across cycles per position
    pos_stds = np.array([np.nanstd(seasonal_index_norm[idx_pos == k]) for k in range(period)])
    return pos_means, pos_stds


def analyze_single_product(
    df_sales_prepared: pd.DataFrame,
    product_id: str,
    product_names: Optional[Dict[str, str]] = None,
    value_col: str = "sales",
    out_dir: str = "analysis_output",
    min_period: int = 2,
    max_period: int = 24,
) -> Tuple[SeasonalityMetrics, Optional[pd.DataFrame]]:
    os.makedirs(out_dir, exist_ok=True)

    product_df = (
        df_sales_prepared[df_sales_prepared["mku"] == product_id]
        .sort_values("date")
        .copy()
    )
    product_name = (product_names or {}).get(product_id, product_id)

    if product_df.empty:
        metrics = SeasonalityMetrics(
            product_id=product_id,
            product_name=product_name,
            n_observations=0,
            detected_period=None,
            candidate_periods="",
            seasonality_strength=None,
            trend_strength=None,
            seasonal_amplitude_abs=None,
            seasonal_amplitude_rel=None,
            seasonal_cv=None,
            seasonal_peak_month=None,
            seasonal_trough_month=None,
            yoy_growth_rate=None,
            cagr_12m=None,
            trend_slope_per_period=None,
            volatility_cv=None,
            concentration_gini=None,
            acf_lag1=None,
            acf_lag_detected=None,
        )
        return metrics, None

    y = _safe_series(product_df[value_col])
    dates = pd.to_datetime(product_df["date"]).dt.to_period("M").dt.to_timestamp()

    # Period detection via ACF peak search
    detected_period, candidate_periods = _infer_period_via_acf(
        y.values, min_period=min_period, max_period=max_period
    )

    # Fall back to common monthly seasonality if not detected
    if detected_period is None and len(y) >= 12:
        detected_period = 12
    elif detected_period is None and len(y) >= 6:
        detected_period = 6

    # Run robust STL if we have at least 2 cycles
    seasonal = trend = remainder = None
    stl_result = None
    if detected_period is not None and len(y) >= 2 * detected_period:
        stl = STL(y.values, period=int(detected_period), robust=True)
        stl_result = stl.fit()
        seasonal = stl_result.seasonal
        trend = stl_result.trend
        remainder = stl_result.resid

    # Strength metrics (Hyndman definitions)
    seasonality_strength = _strength_of_component(seasonal, remainder) if seasonal is not None else None
    trend_strength = _strength_of_component(trend, remainder) if trend is not None else None

    # Seasonal amplitude and CV from seasonal component if available
    seasonal_amplitude_abs = seasonal_amplitude_rel = seasonal_cv = None
    seasonal_peak_month = seasonal_trough_month = None
    if seasonal is not None:
        seasonal_amp = float(np.nanpercentile(seasonal, 95) - np.nanpercentile(seasonal, 5))
        level = float(np.nanmean(y.values)) if np.nanmean(y.values) != 0 else np.nan
        seasonal_amplitude_abs = seasonal_amp
        seasonal_amplitude_rel = float(seasonal_amp / level) if level and not math.isclose(level, 0.0) else np.nan
        # Compute per-position seasonal means across cycles
        pos = np.arange(len(seasonal)) % int(detected_period)
        per_pos_mean = np.array([np.nanmean(seasonal[pos == k]) for k in range(int(detected_period))])
        seasonal_cv = float(np.nanstd(per_pos_mean) / (np.nanmean(np.abs(per_pos_mean)) + 1e-12))
        # Peak/trough timing using average seasonal pattern
        peak_idx = int(np.nanargmax(per_pos_mean))
        trough_idx = int(np.nanargmin(per_pos_mean))
        # Map indices to calendar months based on first date
        if len(dates) > 0:
            first_month = dates.iloc[0].to_period("M")
            peak_period = (first_month + peak_idx).to_timestamp()
            trough_period = (first_month + trough_idx).to_timestamp()
            seasonal_peak_month = peak_period.strftime("%b")
            seasonal_trough_month = trough_period.strftime("%b")

    # YoY growth and CAGR (last 12 months vs previous 12 months)
    yoy_growth_rate = cagr_12m = None
    if len(y) >= 24:
        last12 = float(np.nansum(y.values[-12:]))
        prev12 = float(np.nansum(y.values[-24:-12]))
        if prev12 > 0:
            yoy_growth_rate = float(last12 / prev12 - 1.0)
        else:
            yoy_growth_rate = np.nan
        # CAGR on monthly average scaled to annual
        start_val = float(np.nanmean(y.values[-24:-12]))
        end_val = float(np.nanmean(y.values[-12:]))
        if start_val > 0 and end_val > 0:
            cagr_12m = float((end_val / start_val) ** (1.0 / 1.0) - 1.0)
        else:
            cagr_12m = np.nan

    # Trend slope via Theil–Sen
    trend_slope_per_period = _theil_sen_slope(y.values)

    # Volatility CV (residuals or original if no STL)
    if remainder is not None:
        base = remainder
    else:
        base = y.values - pd.Series(y).rolling(3, min_periods=1, center=True).mean().values
    volatility_cv = float(np.nanstd(base) / (np.nanmean(y.values) + 1e-12))

    # Concentration (Gini) across months (share of sales by month)
    concentration_gini = _compute_gini(y.values)

    # ACF diagnostics
    acf_values = acf(y.values, nlags=min(24, max(2, len(y) // 2)), fft=True)
    acf_lag1 = float(acf_values[1]) if len(acf_values) > 1 else np.nan
    acf_lag_detected = float(acf_values[int(detected_period)]) if (detected_period is not None and int(detected_period) < len(acf_values)) else np.nan

    # Build metrics dataclass
    metrics = SeasonalityMetrics(
        product_id=product_id,
        product_name=product_name,
        n_observations=int(len(y)),
        detected_period=int(detected_period) if detected_period is not None else None,
        candidate_periods=",".join(map(str, candidate_periods)) if candidate_periods else "",
        seasonality_strength=float(seasonality_strength) if seasonality_strength is not None else None,
        trend_strength=float(trend_strength) if trend_strength is not None else None,
        seasonal_amplitude_abs=seasonal_amplitude_abs,
        seasonal_amplitude_rel=seasonal_amplitude_rel,
        seasonal_cv=seasonal_cv,
        seasonal_peak_month=seasonal_peak_month,
        seasonal_trough_month=seasonal_trough_month,
        yoy_growth_rate=yoy_growth_rate,
        cagr_12m=cagr_12m,
        trend_slope_per_period=float(trend_slope_per_period) if trend_slope_per_period is not None else None,
        volatility_cv=volatility_cv,
        concentration_gini=float(concentration_gini) if concentration_gini is not None else None,
        acf_lag1=acf_lag1,
        acf_lag_detected=acf_lag_detected,
    )

    # Plots
    try:
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        safe_name = str(product_name).replace("/", "-").replace(" ", "_")[:60]

        # STL components plot
        if stl_result is not None:
            fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
            axes[0].plot(dates, y.values, label="Original")
            axes[0].set_title(f"{product_name} — Original")
            axes[1].plot(dates, stl_result.trend, color="C1", label="Trend")
            axes[1].set_title("Trend")
            axes[2].plot(dates, stl_result.seasonal, color="C2", label="Seasonal")
            axes[2].set_title("Seasonal")
            axes[3].plot(dates, stl_result.resid, color="C3", label="Remainder")
            axes[3].set_title("Remainder")
            for ax in axes:
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"stl_{safe_name}.png"), dpi=130)
            plt.close(fig)

        # Seasonal indices barplot if period detected
        if metrics.detected_period and len(y) >= metrics.detected_period:
            pos_means, pos_stds = _seasonal_indices(pd.Series(y.values), metrics.detected_period)
            if pos_means.size > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = np.arange(pos_means.size)
                ax.bar(x, pos_means, yerr=pos_stds, alpha=0.8)
                ax.set_title(f"{product_name} — Seasonal Indices (mean=1)")
                ax.set_xlabel("Cycle position")
                ax.set_ylabel("Index")
                ax.grid(True, axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"seasonal_indices_{safe_name}.png"), dpi=130)
                plt.close(fig)

        # Heatmap of monthly values by year (calendar view)
        cal = (
            pd.DataFrame({"date": dates, "value": y.values})
            .assign(year=lambda d: d["date"].dt.year, month=lambda d: d["date"].dt.month)
            .pivot_table(index="year", columns="month", values="value", aggfunc="sum")
            .sort_index()
        )
        if cal.shape[0] >= 1 and cal.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(12, 4 + 0.3 * cal.shape[0]))
            sns.heatmap(cal, annot=False, cmap="Blues", ax=ax)
            ax.set_title(f"{product_name} — Calendar Heatmap (Sum)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"calendar_heatmap_{safe_name}.png"), dpi=130)
            plt.close(fig)

    except Exception:
        # Plotting should not block metrics computation
        pass

    # Return metrics and a small diagnostics DataFrame
    diagnostics = None
    if stl_result is not None:
        diagnostics = pd.DataFrame(
            {
                "date": dates.values,
                "value": y.values,
                "trend": stl_result.trend,
                "seasonal": stl_result.seasonal,
                "remainder": stl_result.resid,
            }
        )

    return metrics, diagnostics


def analyze_products(
    df_sales: pd.DataFrame,
    sample_products: Iterable[str],
    dict_naming: Optional[Dict[str, str]] = None,
    value_cols: List[str] = ("sales", "sales_amount"),
    value_col_for_analysis: str = "sales",
    out_dir: str = "analysis_output",
    min_period: int = 2,
    max_period: int = 24,
) -> pd.DataFrame:
    df_sales_prepared = prepare_dataframe(df_sales, list(value_cols))

    all_metrics: List[SeasonalityMetrics] = []

    for product in sample_products:
        metrics, diagnostics = analyze_single_product(
            df_sales_prepared=df_sales_prepared,
            product_id=product,
            product_names=dict_naming or {},
            value_col=value_col_for_analysis,
            out_dir=out_dir,
            min_period=min_period,
            max_period=max_period,
        )
        all_metrics.append(metrics)

        # Save diagnostics if available
        if diagnostics is not None:
            safe_name = (dict_naming or {}).get(product, product).replace("/", "-").replace(" ", "_")[:60]
            diag_path = os.path.join(out_dir, f"diagnostics_{safe_name}.csv")
            diagnostics.to_csv(diag_path, index=False)

    metrics_df = pd.DataFrame([asdict(m) for m in all_metrics])
    metrics_path = os.path.join(out_dir, "seasonality_metrics.csv")
    os.makedirs(out_dir, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df


__all__ = [
    "prepare_dataframe",
    "analyze_single_product",
    "analyze_products",
    "SeasonalityMetrics",
]