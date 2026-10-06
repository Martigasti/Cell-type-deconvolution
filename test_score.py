import os
import numpy as np
import pandas as pd

import attachement.data_processing as dp
from submission_script import program

EPS = 1e-6


# -----------------------------
# Basic utilities
# -----------------------------
def clip_and_normalize(df, eps=EPS):
    """df: cells x samples"""
    X = df.to_numpy(dtype=float)
    X = np.clip(X, eps, 1.0 - eps)
    col_sums = X.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    X = X / col_sums
    X = np.clip(X, eps, 1.0 - eps)
    X = X / X.sum(axis=0, keepdims=True)
    return pd.DataFrame(X, index=df.index, columns=df.columns)


def clr_transform(P, eps=EPS):
    """
    P: (n_samples x n_cells) compositional
    """
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    gm = np.exp(np.mean(np.log(P), axis=1, keepdims=True))
    return np.log(P / gm)


def aitchison_distance(y_true, y_pred, eps=EPS):
    """
    y_true, y_pred: (n_samples x n_cells)
    returns mean Aitchison distance (lower is better)
    """
    T = clr_transform(y_true, eps=eps)
    P = clr_transform(y_pred, eps=eps)
    d = np.linalg.norm(T - P, axis=1)  # Euclidean in CLR space
    return float(np.mean(d))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -----------------------------
# Correlation metrics
# -----------------------------
def safe_corr_pearson(a, b, eps=1e-12):
    """Pearson correlation between 1D arrays; returns 0 if degenerate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt(np.sum(a * a))
    db = np.sqrt(np.sum(b * b))
    if da < eps or db < eps:
        return 0.0
    return float(np.sum(a * b) / (da * db))


def safe_corr_spearman(a, b):
    """Spearman correlation between 1D arrays; returns 0 if degenerate."""
    a = pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(b, dtype=float)).rank(method="average").to_numpy()
    return safe_corr_pearson(a, b)


def corr_full_matrix(y_true, y_pred, method="pearson"):
    """Correlation on full matrix (flattened)."""
    t = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if method == "pearson":
        return safe_corr_pearson(t, p)
    elif method == "spearman":
        return safe_corr_spearman(t, p)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")


def corr_columns_samples(y_true, y_pred, method="pearson"):
    """
    Column-wise correlations = per-sample correlation across cell types.
    y_* shape: (n_samples x n_cells). For each sample i: corr(y_true[i,:], y_pred[i,:])
    """
    vals = []
    for i in range(y_true.shape[0]):
        if method == "pearson":
            vals.append(safe_corr_pearson(y_true[i, :], y_pred[i, :]))
        else:
            vals.append(safe_corr_spearman(y_true[i, :], y_pred[i, :]))
    return float(np.mean(vals)) if vals else 0.0


def corr_rows_celltypes(y_true, y_pred, method="pearson"):
    """
    Row-wise correlations = per-cell-type correlation across samples.
    y_* shape: (n_samples x n_cells). For each cell j: corr(y_true[:,j], y_pred[:,j])
    """
    vals = []
    for j in range(y_true.shape[1]):
        if method == "pearson":
            vals.append(safe_corr_pearson(y_true[:, j], y_pred[:, j]))
        else:
            vals.append(safe_corr_spearman(y_true[:, j], y_pred[:, j]))
    return float(np.mean(vals)) if vals else 0.0


# -----------------------------
# Scoring mapping (assumptions)
# -----------------------------
def corr_to_score(c):
    """Map correlation [-1,1] -> [0,1]."""
    return float((c + 1.0) / 2.0)


def error_to_score(e):
    """Map error/distance [0,inf) (lower better) -> (0,1] (higher better)."""
    return float(1.0 / (1.0 + e))


def geo_mean(values, eps=1e-15):
    v = np.array(values, dtype=float)
    v = np.clip(v, eps, None)
    return float(np.exp(np.mean(np.log(v))))


def global_score_from_metrics(metrics_dict):
    """
    Implements:
      - correlations block = 1/3
      - errors block = 1/3
      - distance block = 1/3
    Weighted geometric mean, using explicit monotonic mappings.
    """

    # Correlations (6 metrics)
    corr_scores = [
        corr_to_score(metrics_dict["pearson_total"]),
        corr_to_score(metrics_dict["pearson_cols"]),
        corr_to_score(metrics_dict["pearson_rows"]),
        corr_to_score(metrics_dict["spearman_total"]),
        corr_to_score(metrics_dict["spearman_cols"]),
        corr_to_score(metrics_dict["spearman_rows"]),
    ]
    corr_block = geo_mean(corr_scores)

    # Errors (MAE + RMSE)
    err_scores = [
        error_to_score(metrics_dict["mae"]),
        error_to_score(metrics_dict["rmse"]),
    ]
    err_block = geo_mean(err_scores)

    # Distance (Aitchison)
    dist_block = error_to_score(metrics_dict["aitchison"])  # same mapping: lower->higher

    # Equal weights = 1/3 each -> geometric mean across blocks
    global_score = geo_mean([corr_block, err_block, dist_block])

    return corr_block, err_block, dist_block, global_score


# -----------------------------
# Ground truth loading/alignment (unchanged)
# -----------------------------
def load_ground_truth(gt_dir, dataset_key):
    candidates = [
        os.path.join(gt_dir, f"groundtruth_{dataset_key}.h5"),
        os.path.join(gt_dir, f"{dataset_key}.h5"),
        os.path.join(gt_dir, f"groundtruth_{dataset_key}.csv"),
        os.path.join(gt_dir, f"{dataset_key}.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    if path.endswith(".csv"):
        return pd.read_csv(path, index_col=0)

    gt_obj = dp.read_hdf5(path)
    preferred_keys = ["proportions", "prop", "ground_truth", "truth", "y_true", "gt"]
    for k in preferred_keys:
        if k in gt_obj:
            return gt_obj[k]

    best = None
    for k, df in gt_obj.items():
        if isinstance(df, pd.DataFrame) and (df.shape[0] == 5 or df.shape[1] == 5):
            best = df
            break
    if best is None:
        for k, df in gt_obj.items():
            if isinstance(df, pd.DataFrame):
                best = df
                break
    return best


def align_truth_pred(truth_df, pred_df, dataset_key=""):
    def try_align(tdf):
        common_cells = tdf.index.intersection(pred_df.index)
        common_samples = tdf.columns.intersection(pred_df.columns)
        return tdf, common_cells, common_samples

    t_as, cc_as, cs_as = try_align(truth_df)
    t_tr, cc_tr, cs_tr = try_align(truth_df.T)

    if (len(cc_tr) > len(cc_as)) or (len(cc_tr) == len(cc_as) and len(cs_tr) > len(cs_as)):
        truth_use = t_tr
    else:
        truth_use = t_as

    common_cells = truth_use.index.intersection(pred_df.index)
    common_samples = truth_use.columns.intersection(pred_df.columns)

    if len(common_cells) > 0 and len(common_samples) == 0:
        gt_cols = list(truth_use.columns)
        pred_cols = list(pred_df.columns)

        def is_positional_cols(cols):
            n = len(cols)
            try:
                as_int = [int(c) for c in cols]
                return as_int == list(range(n))
            except Exception:
                return False

        if truth_use.shape[1] == pred_df.shape[1] and is_positional_cols(gt_cols):
            truth_use = truth_use.copy()
            truth_use.columns = pred_cols
            common_samples = truth_use.columns.intersection(pred_df.columns)

    common_cells = truth_use.index.intersection(pred_df.index)
    common_samples = truth_use.columns.intersection(pred_df.columns)

    if len(common_cells) == 0 or len(common_samples) == 0:
        print(f"[ALIGN DEBUG] {dataset_key}: still no overlap after fixes.")
        print("  pred cells:", list(pred_df.index))
        print("  truth rows:", list(truth_df.index)[:10], "...")
        print("  truth cols:", list(truth_df.columns)[:10], "...")
        print("  pred samples:", list(pred_df.columns)[:10], "...")
        print("  truth_use shape:", truth_use.shape, "pred shape:", pred_df.shape)

    t = truth_use.loc[common_cells, common_samples]
    p = pred_df.loc[common_cells, common_samples]
    return t, p


# -----------------------------
# Main
# -----------------------------
def main(project_path):
    data_dir = os.path.join(project_path, "data")
    gt_dir = os.path.join(project_path, "ground_truth")

    ref_path = os.path.join(data_dir, "reference_pdac.h5")
    ref = dp.read_hdf5(ref_path)

    mix_files = [f for f in os.listdir(data_dir) if f.startswith("mixes_") and f.endswith(".h5")]
    mix_files.sort()

    rows = []

    for mf in mix_files:
        dataset_key = mf.replace("mixes_", "").replace(".h5", "")
        mix_path = os.path.join(data_dir, mf)
        mixes = dp.read_hdf5(mix_path)

        print(f"\n=== Dataset: {dataset_key} ===")

        pred = program(
            mixes["mix_rna"],
            ref["ref_bulkRNA"],
            mix_met=mixes.get("mix_met", None),
            ref_met=ref.get("ref_met", None),
        )
        pred = clip_and_normalize(pred, eps=EPS)

        truth = None
        if os.path.exists(gt_dir):
            truth = load_ground_truth(gt_dir, dataset_key)

        if truth is None:
            print("[GT] No ground truth found for this dataset. Printing sanity checks only.")
            print("  shape:", pred.shape, "(cells x samples)")
            print("  min/max:", float(pred.min().min()), float(pred.max().max()))
            print("  sum-to-one (first 5):", np.round(pred.sum(axis=0).values[:5], 6))
            continue

        truth, pred_aligned = align_truth_pred(truth, pred, dataset_key=dataset_key)
        truth = clip_and_normalize(truth, eps=EPS)

        # (n_samples x n_cells)
        y_true = truth.T.to_numpy(dtype=float)
        y_pred = pred_aligned.T.to_numpy(dtype=float)

        # Metrics
        m_aitch = aitchison_distance(y_true, y_pred, eps=EPS)
        m_mae = mae(y_true, y_pred)
        m_rmse = rmse(y_true, y_pred)

        p_total = corr_full_matrix(y_true, y_pred, method="pearson")
        p_cols = corr_columns_samples(y_true, y_pred, method="pearson")
        p_rows = corr_rows_celltypes(y_true, y_pred, method="pearson")

        s_total = corr_full_matrix(y_true, y_pred, method="spearman")
        s_cols = corr_columns_samples(y_true, y_pred, method="spearman")
        s_rows = corr_rows_celltypes(y_true, y_pred, method="spearman")

        metrics = {
            "aitchison": m_aitch,
            "mae": m_mae,
            "rmse": m_rmse,
            "pearson_total": p_total,
            "pearson_cols": p_cols,
            "pearson_rows": p_rows,
            "spearman_total": s_total,
            "spearman_cols": s_cols,
            "spearman_rows": s_rows,
        }

        corr_block, err_block, dist_block, gscore = global_score_from_metrics(metrics)

        print(f"  Aitchison (mean, lower=better): {m_aitch:.6f}")
        print(f"  MAE (mean): {m_mae:.6f}")
        print(f"  RMSE: {m_rmse:.6f}")
        print(f"  Pearson total/cols/rows: {p_total:.6f} / {p_cols:.6f} / {p_rows:.6f}")
        print(f"  Spearman total/cols/rows: {s_total:.6f} / {s_cols:.6f} / {s_rows:.6f}")
        print(f"  Block scores (corr/err/dist): {corr_block:.6f} / {err_block:.6f} / {dist_block:.6f}")
        print(f"  >>> Global score: {gscore:.6f}")

        rows.append({
            "dataset": dataset_key,
            "n_samples": y_true.shape[0],
            "aitchison_mean": m_aitch,
            "mae_mean": m_mae,
            "rmse": m_rmse,
            "pearson_total": p_total,
            "pearson_cols": p_cols,
            "pearson_rows": p_rows,
            "spearman_total": s_total,
            "spearman_cols": s_cols,
            "spearman_rows": s_rows,
            "corr_block": corr_block,
            "err_block": err_block,
            "dist_block": dist_block,
            "global_score": gscore,
        })

    if rows:
        summary = pd.DataFrame(rows)

        # Weighted overall global score by n_samples (common approach)
        w = summary["n_samples"].to_numpy(dtype=float)
        overall_global = float(np.average(summary["global_score"], weights=w))

        # Also print a sorted table
        summary_sorted = summary.sort_values("global_score", ascending=False)

        print("\n\n=== Summary (sorted by Global Score, higher=better) ===")
        cols_show = [
            "dataset", "n_samples",
            "global_score",
            "corr_block", "err_block", "dist_block",
            "aitchison_mean", "mae_mean", "rmse",
            "pearson_total", "pearson_cols", "pearson_rows",
            "spearman_total", "spearman_cols", "spearman_rows",
        ]
        print(summary_sorted[cols_show].to_string(index=False))

        print(f"\nFINAL GLOBAL SCORE (weighted by n_samples): {overall_global:.6f}")


def normalize_project_path(p):
    p = os.path.abspath(p)
    if p.endswith(".h5"):
        p = os.path.dirname(os.path.dirname(p))
    return p


if __name__ == "__main__":
    project_path = normalize_project_path(os.getcwd())
    main(project_path)
