"""OJIP Lasso regression pipeline for Supplementary Materials.

Compact single-file script to process AquaPen OJIP per-sheet wide CSVs
found under the repository `CSV/` directory, compute smoothed fluorescence
curves and core OJIP variables, and run a grouped nested Lasso
regression (1-SE rule) to predict NH3(mM). All outputs are written
directly to the repository `Output/` folder.

Getting started:
1. Place your metadata CSV at METADATA_CSV (default: `CSV/metadata.csv`).
2. Ensure the per-sheet wide CSVs (e.g., `Cv1.csv`, `Ao1.csv`) exist under `CSV/`.
3. Run `python ojip_lasso_regression.py`. Results are written to `Output/`.

This minimal script is intentionally direct and assumes well-formed input
for the Supplementary Materials release.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import matplotlib
# Use a non-interactive backend for cross-platform/headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

# --- Global warning filters to keep console clean ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"scipy\.interpolate")
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import NearestNeighbors


def _knn_ad_scores(
    X_train: np.ndarray,
    X_query: np.ndarray | None = None,
    k: int = 5,
    metric: str = 'euclidean',
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return mean kNN distances for train (leave-one-out) and query sets."""
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D array")
    n_train = X_train.shape[0]
    k_eff = max(1, min(k, n_train - 1)) if n_train > 1 else 1
    nn_train = NearestNeighbors(n_neighbors=min(k_eff + 1, n_train), metric=metric)
    nn_train.fit(X_train)
    d_tr, _ = nn_train.kneighbors(X_train)
    tr_scores = d_tr[:, 1:].mean(axis=1) if d_tr.shape[1] > 1 else np.zeros(n_train, dtype=float)
    if X_query is None or n_train == 0:
        return tr_scores, None
    nn_query = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn_query.fit(X_train)
    d_q, _ = nn_query.kneighbors(X_query)
    q_scores = d_q.mean(axis=1) if d_q.size else np.zeros(X_query.shape[0], dtype=float)
    return tr_scores, q_scores


def _compute_ad_knn_active(
    model_lasso,
    scaler: StandardScaler,
    X_tr,
    X_te,
    *,
    ad_quantile: float = 0.975,
    k: int | None = None,
    ad_weight_mode: str = 'none',
    ad_weight_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute AD distances in active-feature space (coefficient-weighted kNN).
    Returns (train_scores, query_scores, threshold).
    """
    cols = getattr(scaler, 'feature_names_in_', None)
    if cols is not None:
        if not isinstance(X_tr, pd.DataFrame):
            X_tr = pd.DataFrame(X_tr, columns=list(cols))
        else:
            try:
                X_tr = X_tr.loc[:, list(cols)]
            except Exception:
                pass
        if not isinstance(X_te, pd.DataFrame):
            X_te = pd.DataFrame(X_te, columns=list(cols))
        else:
            try:
                X_te = X_te.loc[:, list(cols)]
            except Exception:
                pass
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    coef = getattr(model_lasso, 'coef_', None)
    if coef is None:
        act_cols = np.arange(X_tr_s.shape[1])
    else:
        act_cols = np.flatnonzero(np.asarray(coef) != 0)
        if act_cols.size == 0:
            act_cols = np.arange(X_tr_s.shape[1])
    XtrA = X_tr_s[:, act_cols]
    XteA = X_te_s[:, act_cols]
    mode = str(ad_weight_mode or 'none').strip().lower()
    if mode == 'abs_coef' and coef is not None and act_cols.size > 0:
        try:
            w = np.abs(np.asarray(coef, dtype=float).ravel()[act_cols])
            w = np.where(np.isfinite(w), w, 0.0)
            if float(np.sum(w)) > 0:
                w = np.maximum(w, float(ad_weight_eps))
                w = w / (float(np.mean(w)) + float(ad_weight_eps))
                sw = np.sqrt(w)
                XtrA = XtrA * sw
                XteA = XteA * sw
        except Exception:
            pass
    n_tr = XtrA.shape[0]
    if k is None:
        k = max(3, min(10, int(np.sqrt(max(n_tr, 1)))))
        k = min(k, max(1, n_tr - 1))
    tr_scores, te_scores = _knn_ad_scores(XtrA, XteA, k=k)
    thr = float(np.quantile(tr_scores, ad_quantile)) if tr_scores.size else 0.0
    return tr_scores, te_scores, thr

# -----------------------------
# User editable parameters
# -----------------------------

# Point to metadata CSV and the folder containing per-sheet AquaPen OJIP CSVs.
# Resolve paths relative to this script's directory so the script can be run
# from any current working directory and on any OS.
BASE_DIR = Path(__file__).resolve().parent
METADATA_CSV = BASE_DIR / "CSV" / "metadata.csv"
AQUAPEN_OJIP_DIR = BASE_DIR / "CSV"
OUTPUT_DIR = BASE_DIR / "Output"

# (rest of the file preserves the original implementation with only path/doc updates)

FO_TIME_US = 41
K_TIME_US = 301
J_TIME_US = 2_021
I_TIME_US = 30_321
L_TIME_US = 151
P_TIME_US = 301_621
LAST_TIME_US = 2_001_621

REQUIRED_COLUMNS = [
    "Species",
    "Condition",
    "Replicate",
    "TimeLabel",
    "Time",
    "NH3_mM",
    "MeasurementID",
    "OJIPFile",
]

DATA_TEACH: List[Dict[str, str]] = [{"Species": "Cv"}]
DATA_PRED: List[Dict[str, str]] = [{"Species": "Ao"}]

QC_FLAGS_REQUIRE_FALSE: List[str] = [
    "flag_low_FvFm",
    "flag_low_Vk",
    "flag_high_noise_Vj",
]

ALPHA_GRID = np.logspace(-4, 1, 30)

NESTED_OUTER_REPEATS = 3
NESTED_OUTER_FOLDS = 5
NESTED_INNER_FOLDS = 5

GROUP_KEYS: List[str] = ["Species", "Condition", "Replicate"]
STRATIFY_BY: str = "Condition"

PI_METHOD: str = 'outer'
PI_QUANTILE: float = 0.975
MAD_TO_SIGMA: float = 1.4826
PI_MAD_MULTIPLIER: float = 2.24
AD_QUANTILE: float = 0.975
AD_WEIGHT_MODE: str = 'abs_coef'

RAW_LASSO_FEATURES: List[str] = [
    "Vj",
    "Vi",
    "Fm/Fo",
    "Fv/Fo",
    "Fv/Fm",
    "Fp/Fmax",
    "Fo/Fm",
    "Mo",
    "Sm",
    "N",
    "VL",
    "Vk",
    "Vk/Vj",
    "Vj/Vm",
    "Vk/Vm",
    "Vi/Vj",
]


@dataclass
class SampleRecord:
    sample_id: str
    metadata: pd.Series
    time_us: np.ndarray
    fluorescence: np.ndarray
    smooth_us: np.ndarray
    smooth_signal: np.ndarray


def drop_21_31us(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = []
    for key in ('21', '31'):
        try:
            if key in df.index:
                idx.append(key)
        except Exception:
            pass
    return df.drop(index=idx) if idx else df


def _safe_spline_interp(x: np.ndarray, y: np.ndarray, x_out: np.ndarray, s: float | None = None) -> np.ndarray:
    order = np.argsort(x)
    x_s = np.asarray(x, dtype=float)[order]
    y_s = np.asarray(y, dtype=float)[order]
    try:
        spl = UnivariateSpline(x_s, y_s, k=3, s=s)
        res = spl(x_out)
        if np.isnan(res).any():
            res = np.where(np.asarray(x_out) < x_s[0], y_s[0], np.where(np.asarray(x_out) > x_s[-1], y_s[-1], res))
        return res
    except Exception:
        return np.interp(x_out, x_s, y_s, left=y_s[0], right=y_s[-1])


def _aquapen_intervals_us() -> list[tuple[int, int]]:
    return [
        (11, 611),
        (1021, 13921),
        (15321, 90321),
        (101621, 2001621),
    ]


def _fill_aquapen_timegap(start_us: int = 40) -> np.ndarray:
    intervals = _aquapen_intervals_us()
    steps = [10, 100, 1000, 10000]
    out: list[int] = []
    for i, (lo, hi) in enumerate(intervals):
        step = steps[i if i < len(steps) else -1]
        pts = [t for t in range(lo, hi + 1, step) if t >= start_us]
        out.extend(pts)
        if i < len(intervals) - 1:
            next_lo = intervals[i + 1][0]
            gap_start = hi + step
            gap_end = next_lo
            cur = gap_start
            while cur < gap_end:
                out.append(cur)
                cur += step
    return np.asarray(sorted(set(int(v) for v in out)), dtype=float)


def _apply_spline_smoothing(x_us: np.ndarray, y: np.ndarray, *, nknots: int = 30, s: float | None = None) -> np.ndarray:
    if len(x_us) < 4:
        return np.asarray(y, dtype=float).copy()
    order = np.argsort(x_us)
    xs = np.asarray(x_us, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    ux, uid = np.unique(xs, return_index=True)
    xs_u = ux
    ys_u = ys[uid]
    if len(xs_u) < 4:
        try:
            return UnivariateSpline(xs_u, ys_u, k=3, s=s)(x_us)
        except Exception:
            return _safe_spline_interp(x_us, y, x_us, s=s)
    x_min, x_max = float(xs_u[0]), float(xs_u[-1])
    x_min = max(x_min, 1e-8)
    logk = np.linspace(np.log10(x_min), np.log10(x_max), num=int(nknots) + 2)
    knots = 10 ** logk[1:-1]
    try:
        spl = LSQUnivariateSpline(xs_u, ys_u, t=knots, k=3)
        return spl(x_us)
    except Exception:
        try:
            return UnivariateSpline(xs_u, ys_u, k=3, s=s)(x_us)
        except Exception:
            return _safe_spline_interp(x_us, y, x_us, s=s)


def smooth_dataframe(
    df_us: pd.DataFrame,
    strength: int,
    *,
    method: str = 'spline',
    phase_aware: bool = True,
    start_us: int = 40,
    interpolate_gaps: bool = False,
    spline_nknots: int = 30,
):
    if df_us is None or df_us.empty:
        return df_us, df_us
    idx_num = np.array([int(float(i)) for i in df_us.index])
    m_keep = idx_num >= int(start_us)
    df = df_us.loc[m_keep].copy()
    x_us = np.array([int(float(i)) for i in df.index], dtype=float)
    t_us_filled = _fill_aquapen_timegap(start_us)
    out_cols: dict[str, np.ndarray] = {}
    for col in list(df.columns):
        y = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        if method.lower() != 'spline':
            raise ValueError("Only method='spline' is supported in this minimal implementation")
        y_sm = _apply_spline_smoothing(x_us, y, nknots=int(spline_nknots), s=None)
        y_res = _safe_spline_interp(x_us, y_sm, t_us_filled, s=None)
        out_cols[col] = y_res
    out_df = pd.DataFrame(out_cols, index=[str(int(v)) for v in t_us_filled])
    interp_df = df.copy()
    return out_df, interp_df


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")
    df = df[df["OJIPFile"].notna() & df["MeasurementID"].notna()].copy()
    df["Sample_ID"] = df.apply(
        lambda row: f"{row['Species']}_{row['Condition']}_R{row['Replicate']}_{row['TimeLabel']}_MID{int(row['MeasurementID'])}@{Path(str(row['OJIPFile'])).name}",
        axis=1,
    )
    return df


def load_ojip_from_wide(csv_path: Path, measurement_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"OJIP file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.columns[0] != "index":
        raise ValueError(f"{csv_path}: first column must be 'index'")
    if df.iloc[0, 0] != "index" or df.iloc[1, 0] != "time" or df.iloc[2, 0] != "id":
        raise ValueError(f"{csv_path}: missing header rows ('index', 'time', 'id')")
    col = str(int(measurement_id))
    if col not in df.columns:
        raise KeyError(f"MeasurementID {measurement_id} not in {csv_path.name}")
    if str(df.loc[2, col]) != "OJIP":
        raise ValueError(f"MeasurementID {measurement_id} in {csv_path.name} is not an OJIP column")
    t = pd.to_numeric(df.iloc[3:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.loc[3:, col], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    order = np.argsort(t)
    return t[order].astype(float), y[order].astype(float)


def _nearest_index(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - float(target))))


def _value_at(time_us: np.ndarray, signal: np.ndarray, target_us: float) -> float:
    if time_us.size == 0 or signal.size == 0:
        return float('nan')
    idx = _nearest_index(time_us, target_us)
    return float(signal[idx])


def compute_ojip_variables(time_us: np.ndarray, signal: np.ndarray,
                           time_raw: np.ndarray | None = None,
                           signal_raw: np.ndarray | None = None) -> Dict[str, float]:
    fo = _value_at(time_us, signal, FO_TIME_US)
    fm_idx = int(np.argmax(signal))
    fm = float(signal[fm_idx])
    fv = fm - fo
    if fv == 0:
        fv = np.nan
    vt = (signal - fo) / fv if np.isfinite(fv) and fv != 0 else np.full_like(signal, np.nan)
    time_ms = time_us / 1000.0
    t_fm = time_us[fm_idx]

    fk = _value_at(time_us, signal, K_TIME_US)
    fj = _value_at(time_us, signal, J_TIME_US)
    fi = _value_at(time_us, signal, I_TIME_US)
    fl = _value_at(time_us, signal, L_TIME_US)
    fp = _value_at(time_us, signal, P_TIME_US)
    flast = _value_at(time_us, signal, LAST_TIME_US)

    vk = (fk - fo) / fv if fv else np.nan
    vj = (fj - fo) / fv if fv else np.nan
    vi = (fi - fo) / fv if fv else np.nan
    vl = (fl - fo) / fv if fv else np.nan
    vlast = (flast - fo) / fv if fv else np.nan
    vk_over_vj = vk / vj if vj not in (0, np.nan) else np.nan
    vi_over_vj = vi / vj if vj not in (0, np.nan) else np.nan
    vj_over_vm = vj / (fm - fo) if (fm - fo) != 0 else np.nan
    vk_over_vm = vk / (fm - fo) if (fm - fo) != 0 else np.nan

    fm_over_fo = fm / fo if fo else np.nan
    fv_over_fo = fv / fo if fo else np.nan
    fv_over_fm = fv / fm if fm else np.nan
    fo_over_fm = fo / fm if fm else np.nan

    local_min = float(np.min(signal[max(fm_idx - 5, 0) : fm_idx + 1]))
    fp_over_fmax = (fp - local_min) / (fm - local_min) if fm != local_min else np.nan

    m0 = 4.0 * (fk - fo) / fv if fv else np.nan
    mask = time_us <= t_fm
    area = float(integrate.trapezoid(fm - signal[mask], time_ms[mask])) if np.any(mask) else np.nan
    sm = area / (fm - fo) if fv else np.nan
    n = m0 * sm if np.isfinite(m0) and np.isfinite(sm) else np.nan

    phi_po = fv_over_fm
    psi_eo = 1 - vj if np.isfinite(vj) else np.nan
    phi_eo = phi_po * psi_eo if np.isfinite(phi_po) and np.isfinite(psi_eo) else np.nan
    phi_do = 1 - phi_po if np.isfinite(phi_po) else np.nan
    phi_pav = (phi_po + phi_eo) / 2 if np.isfinite(phi_po) and np.isfinite(phi_eo) else np.nan

    abs_rc = (phi_po / psi_eo) if np.isfinite(phi_po) and np.isfinite(psi_eo) and psi_eo not in (0, np.nan) else np.nan
    tro_rc = phi_po * abs_rc if np.isfinite(abs_rc) and np.isfinite(phi_po) else np.nan
    eto_rc = psi_eo * tro_rc if np.isfinite(psi_eo) and np.isfinite(tro_rc) else np.nan
    dio_rc = abs_rc - tro_rc if np.isfinite(abs_rc) and np.isfinite(tro_rc) else np.nan

    if np.isfinite(phi_po) and phi_po not in (0, 1) and np.isfinite(psi_eo) and psi_eo not in (0, 1):
        piabs = (phi_po / (1 - phi_po)) * (psi_eo / (1 - psi_eo))
    else:
        piabs = np.nan

    ptime = t_fm / 1000.0
    bav = 1 - (sm / ptime) if np.isfinite(sm) and ptime else np.nan

    out = {
        "Fo": fo,
        "Fm": fm,
        "Fm/Fo": fm_over_fo,
        "Fv/Fo": fv_over_fo,
        "Fv/Fm": fv_over_fm,
        "Fv/Fm (phiPo)": fv_over_fm,
        "Vk": vk,
        "Vj": vj,
        "Vi": vi,
        "VL": vl,
        "Vlast": vlast,
        "Vk/Vj": vk_over_vj,
        "Vj/Vm": vj_over_vm,
        "Vk/Vm": vk_over_vm,
        "Vi/Vj": vi_over_vj,
        "Fo/Fm": fo_over_fm,
        "Fp/Fmax": fp_over_fmax,
        "Mo": m0,
        "Sm": sm,
        "N": n,
        "phiPo": phi_po,
        "psiEo": psi_eo,
        "phiEo": phi_eo,
        "phiDo": phi_do,
        "phiPav": phi_pav,
        "ABS/RC": abs_rc,
        "TRo/RC": tro_rc,
        "ETo/RC": eto_rc,
        "DIo/RC": dio_rc,
        "PIabs": piabs,
        "Ptime": ptime,
        "Bav": bav,
    }

    out["flag_low_Fo"] = float(fo < 300)
    out["flag_low_FvFm"] = float(np.isfinite(fv_over_fm) and (fv_over_fm < 0.05))
    out["flag_low_Vk"] = float(np.isfinite(out["Vk"]) and (out["Vk"] < 0.10))
    flag_high_noise_vj = 0.0
    if time_raw is not None and signal_raw is not None and len(time_raw) >= 1:
        try:
            t = np.asarray(time_raw, dtype=float)
            y = np.asarray(signal_raw, dtype=float)
            n = len(y)
            j = int(np.argmin(np.abs(t - 2021.0)))
            i_fo = int(np.argmin(np.abs(t - 41.0)))
            fj_raw = float(y[j])
            fo_raw = float(y[i_fo])
            a = max(0, j - 3)
            b = min(n, j + 2)
            if b > a:
                std_vj = float(np.nanstd(y[a:b]))
                denom = fj_raw - fo_raw
                if np.isfinite(denom) and denom != 0.0 and np.isfinite(std_vj):
                    ratio = abs(std_vj / denom)
                    flag_high_noise_vj = float(ratio > 0.15)
        except Exception:
            flag_high_noise_vj = 0.0
    out["flag_high_noise_Vj"] = flag_high_noise_vj
    return out


def _drop_na_target(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta[meta["NH3_mM"].notna()].copy()
    if out.empty:
        raise ValueError("No rows with NH3_mM target available after filtering metadata")
    return out


def _select_alpha_1se(alpha_grid: np.ndarray, mean_rmse: np.ndarray, se_rmse: np.ndarray) -> float:
    i_min = int(np.argmin(mean_rmse))
    threshold = float(mean_rmse[i_min] + se_rmse[i_min])
    ok = np.where(mean_rmse <= threshold)[0]
    return float(alpha_grid[int(ok[-1])]) if ok.size else float(alpha_grid[i_min])


def _build_groups_and_strata(meta: pd.DataFrame, group_keys: List[str], strat_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not all(k in meta.columns for k in group_keys):
        raise ValueError(f"Missing group keys in metadata: {group_keys}")
    if strat_key and strat_key not in meta.columns:
        raise ValueError(f"Stratify key '{strat_key}' not found in metadata")
    grp = meta[group_keys].astype(str).agg('|'.join, axis=1).to_numpy()
    strat = meta[strat_key].astype(str).to_numpy() if strat_key else np.array(["_"] * len(meta))
    return grp, strat


def _stratified_group_kfold_indices(group_ids: np.ndarray, strat_labels: np.ndarray, *, n_splits: int, random_state: int = 0):
    df = pd.DataFrame({"group": group_ids, "strat": strat_labels})
    gtab = df.groupby("group")["strat"].first().reset_index()
    uniq_groups = gtab["group"].to_numpy()
    uniq_strat = gtab["strat"].astype(str).to_numpy()
    n_groups = len(uniq_groups)
    class_counts = pd.Series(uniq_strat).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    n_splits_eff = min(n_splits, n_groups, min_class_count)
    if n_splits_eff < 2:
        rng = np.random.RandomState(random_state)
        order = rng.permutation(n_groups)
        n_test = max(1, int(round(0.2 * n_groups))) if n_groups >= 2 else 1
        te_g = set(uniq_groups[order[:n_test]])
        tr_g = set(uniq_groups) - te_g
        group_to_indices = {gid: np.where(group_ids == gid)[0] for gid in uniq_groups}
        tr_idx = np.concatenate([group_to_indices[g] for g in tr_g]) if len(tr_g) else np.array([], dtype=int)
        te_idx = np.concatenate([group_to_indices[g] for g in te_g]) if len(te_g) else np.array([], dtype=int)
        yield np.sort(tr_idx), np.sort(te_idx)
        return
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    group_to_indices = {}
    for gid in uniq_groups:
        group_to_indices[gid] = np.where(group_ids == gid)[0]
    for g_tr_idx, g_te_idx in skf.split(np.zeros(n_groups), uniq_strat):
        tr_groups = set(uniq_groups[g_tr_idx])
        te_groups = set(uniq_groups[g_te_idx])
        tr_idx = np.concatenate([group_to_indices[g] for g in tr_groups])
        te_idx = np.concatenate([group_to_indices[g] for g in te_groups])
        yield np.sort(tr_idx), np.sort(te_idx)


def _inner_cv_errors_lasso_grouped(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                                   group_ids: np.ndarray, strat_labels: np.ndarray,
                                   *, inner_folds: int = 5, random_state: int = 0):
    mean_rmse = []
    se_rmse = []
    all_residuals = []
    for a in alphas:
        errs = []
        for tr_idx, va_idx in _stratified_group_kfold_indices(group_ids, strat_labels, n_splits=inner_folds, random_state=random_state):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            base = Pipeline([
                ("scale", StandardScaler()),
                ("lasso", Lasso(alpha=float(a), max_iter=20000, random_state=random_state)),
            ])
            model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
            model.fit(X_tr, y_tr)
            yhat_va = model.predict(X_va)
            err = float(np.sqrt(np.mean((yhat_va - y_va) ** 2)))
            errs.append(err)
            all_residuals.append(y_va - yhat_va)
        errs = np.array(errs, dtype=float)
        mean_rmse.append(float(np.mean(errs)))
        se_rmse.append(float(errs.std(ddof=1) / np.sqrt(len(errs))))
    return np.array(mean_rmse, dtype=float), np.array(se_rmse, dtype=float), (np.concatenate(all_residuals) if all_residuals else np.array([], dtype=float))


def nested_lasso_teach_predictions_grouped(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                                           *, group_keys: List[str], strat_key: str,
                                           repeats: int = 3, outer_folds: int = 5, inner_folds: int = 5,
                                           random_state: int = 0):
    n = len(y)
    y_oof_sum = np.zeros(n, dtype=float)
    y_oof_counts = np.zeros(n, dtype=int)
    all_alphas: List[float] = []
    val_residuals_all = []

    grp_ids_all, strat_all = _build_groups_and_strata(meta.reset_index(drop=True), group_keys, strat_key)

    for r in range(repeats):
        seed = int(random_state + r)
        for tr_idx, te_idx in _stratified_group_kfold_indices(grp_ids_all, strat_all, n_splits=outer_folds, random_state=seed):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            grp_tr = grp_ids_all[tr_idx]
            strat_tr = strat_all[tr_idx]
            mean_rmse, se_rmse, inner_res = _inner_cv_errors_lasso_grouped(
                X_tr, y_tr, ALPHA_GRID, grp_tr, strat_tr,
                inner_folds=inner_folds, random_state=seed
            )
            a_star = _select_alpha_1se(ALPHA_GRID, mean_rmse, se_rmse)
            all_alphas.append(a_star)
            base = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=float(a_star), max_iter=20000, random_state=seed))])
            model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            y_oof_sum[te_idx] += yhat
            y_oof_counts[te_idx] += 1
            val_residuals_all.append(inner_res)

    with np.errstate(invalid='ignore'):
        y_oof_mean = np.divide(y_oof_sum, np.maximum(1, y_oof_counts))
    if val_residuals_all:
        res_all = np.concatenate(val_residuals_all)
        abs_res = np.abs(res_all)
        pi_abs_q95 = float(np.quantile(abs_res, 0.95))
        med = float(np.median(res_all))
        mad = float(np.median(np.abs(res_all - med)))
        sigma = MAD_TO_SIGMA * mad
        pi_mad97_half = float(PI_MAD_MULTIPLIER * sigma)
    else:
        pi_abs_q95 = np.nan
        pi_mad97_half = np.nan
    return y_oof_mean, all_alphas, pi_abs_q95, pi_mad97_half


def _select_by_queries(df: pd.DataFrame, queries: List[Dict[str, str]]) -> pd.DataFrame:
    if not queries:
        return df.copy()
    mask = pd.Series(False, index=df.index)
    for q in queries:
        m = pd.Series(True, index=df.index)
        for k, v in q.items():
            m &= df[k].astype(str) == str(v)
        mask |= m
    return df[mask].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": int(len(yt))}



def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_CSV)
    metadata = _drop_na_target(metadata)
    if 'Condition' in metadata.columns:
        metadata = metadata[metadata['Condition'].astype(str).str.casefold() != 'water'].copy()

    records: List[SampleRecord] = []
    for _, row in metadata.iterrows():
        csv_path = AQUAPEN_OJIP_DIR / str(row["OJIPFile"])
        time_raw, signal_raw = load_ojip_from_wide(csv_path, int(row["MeasurementID"]))
        if time_raw.size > 0:
            try:
                i11 = _nearest_index(time_raw, 11.0)
                bckg = float(signal_raw[i11])
            except Exception:
                bckg = 0.0
        else:
            bckg = 0.0
        signal_bg = signal_raw - float(bckg)
        records.append(
            SampleRecord(
                sample_id=row["Sample_ID"],
                metadata=row,
                time_us=time_raw,
                fluorescence=signal_bg,
                smooth_us=np.array([]),
                smooth_signal=np.array([]),
            )
        )

    smoothed_df = None
    for rec in records:
        df_single = pd.DataFrame({rec.sample_id: rec.fluorescence}, index=[str(int(v)) for v in rec.time_us])
        df_single = drop_21_31us(df_single)
        df_sm, _ = smooth_dataframe(
            df_single,
            strength=30,
            method='spline',
            phase_aware=True,
            start_us=40,
            interpolate_gaps=False,
            spline_nknots=30,
        )
        idx_us = np.array([float(x) for x in df_sm.index], dtype=float)
        rec.smooth_us = idx_us
        rec.smooth_signal = pd.to_numeric(df_sm[rec.sample_id], errors='coerce').to_numpy(dtype=float)
        if smoothed_df is None:
            smoothed_df = pd.DataFrame(index=df_sm.index)
        smoothed_df[rec.sample_id] = df_sm[rec.sample_id]

    col_order = metadata["Sample_ID"].tolist()
    smoothed_df = smoothed_df.reindex(columns=col_order)
    smoothed_df.to_csv(OUTPUT_DIR / "smoothed_curves.csv")

    variables = {}
    for rec in records:
        vars_rec = compute_ojip_variables(rec.smooth_us, rec.smooth_signal, time_raw=rec.time_us, signal_raw=rec.fluorescence)
        variables[rec.sample_id] = vars_rec
    variables_df = pd.DataFrame.from_dict(variables, orient="index")
    variables_df.index.name = "Sample_ID"
    variables_df = variables_df.loc[col_order]
    variables_df.to_csv(OUTPUT_DIR / "ojip_variables.csv")

    meta_teach = _select_by_queries(metadata, DATA_TEACH)
    meta_pred = _select_by_queries(metadata, DATA_PRED)
    if meta_teach.empty:
        meta_teach = metadata.copy()
    if meta_pred.empty:
        meta_pred = metadata.copy()

    cols = [c for c in QC_FLAGS_REQUIRE_FALSE if c in variables_df.columns]
    qc_mask = (variables_df[cols] == 0).all(axis=1) if cols else pd.Series(True, index=variables_df.index)
    good_ids = variables_df.index[qc_mask]
    meta_teach = meta_teach[meta_teach["Sample_ID"].isin(good_ids)].copy()
    meta_pred = meta_pred[meta_pred["Sample_ID"].isin(good_ids)].copy()

    feature_cols = [col for col in RAW_LASSO_FEATURES if col in variables_df.columns]
    missing_features = [col for col in RAW_LASSO_FEATURES if col not in variables_df.columns]
    if missing_features:
        raise ValueError(f"Missing required raw feature columns: {missing_features}")
    X_teach = variables_df.loc[meta_teach["Sample_ID"], feature_cols].to_numpy()
    y_teach = meta_teach["NH3_mM"].to_numpy(dtype=float)
    X_pred = variables_df.loc[meta_pred["Sample_ID"], feature_cols].to_numpy()
    y_pred_true = meta_pred["NH3_mM"].to_numpy(dtype=float)

    y_teach_pred_cv, chosen_alphas, pi_abs_q95, pi_mad97_half = nested_lasso_teach_predictions_grouped(
        X_teach, y_teach, meta_teach,
        group_keys=GROUP_KEYS, strat_key=STRATIFY_BY,
        repeats=NESTED_OUTER_REPEATS, outer_folds=NESTED_OUTER_FOLDS, inner_folds=NESTED_INNER_FOLDS,
        random_state=0,
    )
    m_teach = _metrics(y_teach, y_teach_pred_cv)

    alpha_final = float(np.median(chosen_alphas)) if len(chosen_alphas) > 0 else 1.0
    base = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=alpha_final, max_iter=20000, random_state=0))])
    final_model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    final_model.fit(X_teach, y_teach)
    y_pred_pred = final_model.predict(X_pred)
    m_pred = _metrics(y_pred_true, y_pred_pred)

    res_teach_abs = np.abs(y_teach - y_teach_pred_cv)
    pi_half_mad97 = float(pi_mad97_half) if isinstance(pi_mad97_half, (int, float)) and np.isfinite(pi_mad97_half) and pi_mad97_half > 0 else np.nan
    pi_half_q95 = float(pi_abs_q95) if isinstance(pi_abs_q95, (int, float)) and np.isfinite(pi_abs_q95) and pi_abs_q95 > 0 else np.nan
    pi_half_oof_q95 = float(np.quantile(res_teach_abs, PI_QUANTILE)) if np.isfinite(res_teach_abs).any() else np.nan

    method = (PI_METHOD or '').lower()
    if method == 'quantile':
        pi_width = next((v for v in [pi_half_q95, pi_half_oof_q95] if isinstance(v, (int, float)) and np.isfinite(v) and v > 0), 0.5)
        src = 'inner_abs_q95' if np.isfinite(pi_half_q95) and pi_half_q95 > 0 else 'oof_abs_q95'
        chosen_val = float(pi_width)
        pi_label_pct = int(PI_QUANTILE * 100)
    elif method == 'outer':
        pi_width = next((v for v in [pi_half_oof_q95, pi_half_q95, pi_half_mad97] if isinstance(v, (int, float)) and np.isfinite(v) and v > 0), 0.5)
        src = 'oof_abs_q' if np.isfinite(pi_half_oof_q95) and pi_half_oof_q95 > 0 else ('inner_abs_q95' if np.isfinite(pi_half_q95) and pi_half_q95 > 0 else 'inner_mad97')
        chosen_val = float(pi_width)
        pi_label_pct = int(PI_QUANTILE * 100)
    else:
        pi_width = next((v for v in [pi_half_mad97, pi_half_oof_q95] if isinstance(v, (int, float)) and np.isfinite(v) and v > 0), 0.5)
        src = 'inner_mad97' if np.isfinite(pi_half_mad97) and pi_half_mad97 > 0 else 'oof_abs_q95'
        chosen_val = float(pi_width)
        pi_label_pct = 97.5
    print(f"[PI] Selected half-width source: {src}; value = {chosen_val:.6g} (±{pi_label_pct}% PI)")

    pipe_plain = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=alpha_final, max_iter=20000, random_state=0))])
    pipe_plain.fit(X_teach, y_teach)
    scaler: StandardScaler = pipe_plain.named_steps["scale"]
    lasso: Lasso = pipe_plain.named_steps["lasso"]
    Xtr_s = scaler.transform(X_teach)
    Xpr_s = scaler.transform(X_pred)
    coef = np.asarray(lasso.coef_).ravel()
    try:
        df_coef = pd.DataFrame({
            'feature': feature_cols,
            'coef_std': coef,
        })
        df_coef['abs_coef_std'] = df_coef['coef_std'].abs()
        df_coef['selected'] = df_coef['coef_std'] != 0
        df_coef.sort_values('abs_coef_std', ascending=False, inplace=True)
        df_coef.to_csv(OUTPUT_DIR / 'selected_features_coefficients.csv', index=False)
        print(f"[Coefficients] Saved {int(df_coef['selected'].sum())} selected / {len(df_coef)} total features to 'selected_features_coefficients.csv'")
    except Exception as _e:
        print(f"[Coefficients] Failed to save coefficients CSV: {_e}")
    tr_scores, pr_scores, thr = _compute_ad_knn_active(
        lasso,
        scaler,
        X_teach,
        X_pred,
        ad_quantile=AD_QUANTILE,
        ad_weight_mode=AD_WEIGHT_MODE,
    )
    ad_out_teach = np.zeros(len(y_teach), dtype=bool)
    ad_out_pred = np.asarray(pr_scores, dtype=float) > float(thr) if pr_scores is not None else np.array([False] * len(y_pred_true))
    active_features = int(np.count_nonzero(np.asarray(coef, dtype=float)))
    print(f"[AD] weight_mode={AD_WEIGHT_MODE}; threshold={thr:.6g}; active_features={active_features}")

    conds = pd.concat([meta_teach["Condition"], meta_pred["Condition"]]).astype(str).unique().tolist()
    colors = plt.get_cmap("tab10", max(10, len(conds)))
    markers = ['o','s','^','D','v','P','X','*','<','>']
    cond_to_color = {c: colors(i % colors.N) for i, c in enumerate(conds)}
    cond_to_marker = {c: markers[i % len(markers)] for i, c in enumerate(conds)}

    def _scatter_panel(ax, Xpred, Yobs, meta, ad_out_mask, title, pi_half):
        for i, (yp, yo, cond) in enumerate(zip(Xpred, Yobs, meta["Condition"].astype(str))):
            ax.scatter(yp, yo,
                       color=cond_to_color[cond],
                       marker=cond_to_marker[cond],
                       edgecolor='white', linewidth=0.6, s=32)
        idx = np.where(np.asarray(ad_out_mask, dtype=bool))[0] if ad_out_mask is not None else np.array([], dtype=int)
        if len(idx) > 0:
            ax.scatter(np.array(Xpred)[idx], np.array(Yobs)[idx],
                       color='none', edgecolor='black', marker='o', linewidth=1.2, s=60, facecolors='none', label='AD-out')
        x_all = np.array(Xpred)
        y_all = np.array(Yobs)
        lo = float(np.nanmin(np.concatenate([x_all, y_all])))
        hi = float(np.nanmax(np.concatenate([x_all, y_all])))
        pad = 0.05 * (hi - lo) if np.isfinite(hi - lo) else 0
        lo -= pad; hi += pad
        xx = np.linspace(lo, hi, 100)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.0)
        if np.isfinite(pi_half) and pi_half > 0:
            ax.plot(xx, xx + pi_half, color='red', linestyle='--', linewidth=1.0)
            ax.plot(xx, xx - pi_half, color='red', linestyle='--', linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Predicted NH3(mM)")
        ax.set_ylabel("Observed NH3(mM)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        handles = []
        labels = []
        for cond in conds:
            h = plt.Line2D([0],[0], marker=cond_to_marker[cond], color='w', markerfacecolor=cond_to_color[cond],
                           markeredgecolor='white', markersize=7, linestyle='')
            handles.append(h)
            labels.append(cond)
        if len(idx) > 0:
            handles.append(plt.Line2D([0],[0], marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='', markersize=7))
            labels.append('AD-out')
        ax.legend(handles, labels, frameon=True, framealpha=0.9, fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    _scatter_panel(axes[0], y_teach_pred_cv, y_teach, meta_teach, None,
                   f"Teaching (nested Lasso CV, \u03b1~{alpha_final:.6g}; PI±{pi_width:.2f})", pi_width)
    _scatter_panel(axes[1], y_pred_pred, y_pred_true, meta_pred, ad_out_pred,
                   f"Prediction (\u03b1={alpha_final:.6g}; PI±{pi_width:.2f})", pi_width)

    def _metrics_text(m: Dict[str, float], n_ad: int) -> str:
        return f"R2={m['R2']:.2f}\nRMSE={m['RMSE']:.2f}\nMAE={m['MAE']:.2f}\nn={m['n']}\nAD-out={n_ad}"
    axes[0].text(0.02, 0.98, _metrics_text(m_teach, int(np.sum(ad_out_teach))), transform=axes[0].transAxes,
                 va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(0.02, 0.98, _metrics_text(m_pred, int(np.sum(ad_out_pred))), transform=axes[1].transAxes,
                 va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nh3_predicted_vs_observed_teach_pred.png", dpi=300)
    plt.close(fig)

    combined = variables_df.join(metadata.set_index("Sample_ID"), how="left")
    combined.to_csv(OUTPUT_DIR / "ojip_variables_with_metadata.csv")


if __name__ == "__main__":
    run_pipeline()
