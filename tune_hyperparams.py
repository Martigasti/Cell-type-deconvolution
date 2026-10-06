# tune_hyperparams.py
#
# Random hyperparameter search for submission_script.program(), inspired by test_score.py
# - Evaluates each random config on ALL datasets with available ground truth
# - Tracks per-dataset metrics + weighted FINAL GLOBAL SCORE
# - Prints best config + summarized stats per dataset
#
# Usage:
#   python3 tune_hyperparams.py --trials 50 --seed 42
#   python3 tune_hyperparams.py --trials 100 --seed 1 --show-top 5
#
# Notes:
# - This script calls `program()` directly (no zip export).
# - It loads data/ground_truth once, then runs trials.
# - It uses FINAL GLOBAL SCORE (weighted by n_samples) as the objective.

import os
import json
import math
import argparse
import numpy as np
import pandas as pd

import attachement.data_processing as dp
from submission_script import program

EPS = 1e-6


# -----------------------------
# Basic utilities (same as test_score.py)
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
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    gm = np.exp(np.mean(np.log(P), axis=1, keepdims=True))
    return np.log(P / gm)


def aitchison_distance(y_true, y_pred, eps=EPS):
    T = clr_transform(y_true, eps=eps)
    P = clr_transform(y_pred, eps=eps)
    d = np.linalg.norm(T - P, axis=1)
    return float(np.mean(d))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -----------------------------
# Correlation metrics
# -----------------------------
def safe_corr_pearson(a, b, eps=1e-12):
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
    a = pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(b, dtype=float)).rank(method="average").to_numpy()
    return safe_corr_pearson(a, b)


def corr_full_matrix(y_true, y_pred, method="pearson"):
    t = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if method == "pearson":
        return safe_corr_pearson(t, p)
    if method == "spearman":
        return safe_corr_spearman(t, p)
    raise ValueError("method must be 'pearson' or 'spearman'")


def corr_columns_samples(y_true, y_pred, method="pearson"):
    vals = []
    for i in range(y_true.shape[0]):
        vals.append(
            safe_corr_pearson(y_true[i, :], y_pred[i, :])
            if method == "pearson"
            else safe_corr_spearman(y_true[i, :], y_pred[i, :])
        )
    return float(np.mean(vals)) if vals else 0.0


def corr_rows_celltypes(y_true, y_pred, method="pearson"):
    vals = []
    for j in range(y_true.shape[1]):
        vals.append(
            safe_corr_pearson(y_true[:, j], y_pred[:, j])
            if method == "pearson"
            else safe_corr_spearman(y_true[:, j], y_pred[:, j])
        )
    return float(np.mean(vals)) if vals else 0.0


# -----------------------------
# Scoring mapping
# -----------------------------
def corr_to_score(c):
    return float((c + 1.0) / 2.0)


def error_to_score(e):
    return float(1.0 / (1.0 + e))


def geo_mean(values, eps=1e-15):
    v = np.array(values, dtype=float)
    v = np.clip(v, eps, None)
    return float(np.exp(np.mean(np.log(v))))


def global_score_from_metrics(metrics_dict):
    corr_scores = [
        corr_to_score(metrics_dict["pearson_total"]),
        corr_to_score(metrics_dict["pearson_cols"]),
        corr_to_score(metrics_dict["pearson_rows"]),
        corr_to_score(metrics_dict["spearman_total"]),
        corr_to_score(metrics_dict["spearman_cols"]),
        corr_to_score(metrics_dict["spearman_rows"]),
    ]
    corr_block = geo_mean(corr_scores)

    err_scores = [
        error_to_score(metrics_dict["mae"]),
        error_to_score(metrics_dict["rmse"]),
    ]
    err_block = geo_mean(err_scores)

    dist_block = error_to_score(metrics_dict["aitchison"])

    global_score = geo_mean([corr_block, err_block, dist_block])
    return corr_block, err_block, dist_block, global_score


# -----------------------------
# Ground truth loading/alignment
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
    for _, df in gt_obj.items():
        if isinstance(df, pd.DataFrame) and (df.shape[0] == 5 or df.shape[1] == 5):
            best = df
            break
    if best is None:
        for _, df in gt_obj.items():
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

    t = truth_use.loc[common_cells, common_samples]
    p = pred_df.loc[common_cells, common_samples]
    return t, p


# -----------------------------
# Random search utilities
# -----------------------------
def log_uniform(rng, a, b):
    la, lb = math.log10(a), math.log10(b)
    return float(10 ** (la + (lb - la) * rng.random()))


def clamp(x, lo, hi):
    return float(min(max(x, lo), hi))


def jitter_int_choice(rng, value, choices, max_step=1):
    """
    Move to a nearby option in `choices` (sorted), up to +/- max_step indices.
    """
    choices = sorted(list(choices))
    if value not in choices:
        # snap to closest
        value = min(choices, key=lambda c: abs(c - value))
    i = choices.index(value)
    j = int(np.clip(i + rng.integers(-max_step, max_step + 1), 0, len(choices) - 1))
    return int(choices[j])


# --- BEST CONFIG as the center for local search (minimal change, big impact) ---
BEST_CFG_CENTER = {
    "top_k_rna": 5000,
    "per_cell_k_rna": 800,
    "top_k_met": 3000,
    "per_cell_k_met": 1600,
    "lam_rna": 1.8105176668544015e-05,
    "lam_met": 0.0022362968362077754,
    "tau_rna": 8.466667222956957e-05,
    "tau_met": 1.5603297353262894e-07,
    "do_clr_refine": True,
    "clr_lam2_rna": 9.474724399298411e-05,
    "clr_lam2_met": 2.9068786176483426e-06,
    "clr_iters": 20,
    "fusion_temp": 0.3,
    "alpha_min": 0.05,
    "alpha_max": 0.75,
    "use_unknown": True,
    "unknown_weight": 0.3,
    "scale_kind_rna": "mean",
    "scale_kind_met": "mean",
}


def sample_config_broad(rng):
    # (your original sampler unchanged)
    top_k_rna = int(rng.choice([1200, 1500, 2000, 2500, 3200, 4000, 5000, 6500, 8000]))
    per_cell_k_rna = int(rng.choice([150, 250, 400, 600, 800, 1000, 1200, 1500]))

    top_k_met = int(rng.choice([2000, 3000, 4000, 6000, 8000, 10000, 12000]))
    per_cell_k_met = int(rng.choice([200, 300, 500, 800, 1200, 1600, 2000]))

    lam_rna = log_uniform(rng, 1e-6, 1e-2)
    lam_met = log_uniform(rng, 1e-6, 1e-2)

    clr_lam2_rna = log_uniform(rng, 1e-6, 1e-2)
    clr_lam2_met = log_uniform(rng, 1e-6, 1e-2)

    clr_iters = int(rng.choice([20, 30, 40, 60, 80, 120]))

    tau_rna = log_uniform(rng, 1e-7, 3e-4)
    tau_met = log_uniform(rng, 1e-7, 3e-4)

    fusion_temp = float(rng.choice([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]))
    alpha_min = float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.30, 0.35]))
    alpha_max = float(rng.choice([0.65, 0.75, 0.80, 0.90, 0.95]))
    if alpha_min >= alpha_max:
        alpha_min, alpha_max = 0.2, 0.8

    use_unknown = bool(rng.choice([False, True]))
    unknown_weight = float(rng.choice([0.20, 0.30, 0.35, 0.45, 0.60]))

    scale_kind_rna = str(rng.choice(["mean", "median"]))
    scale_kind_met = str(rng.choice(["mean", "median"]))

    return {
        "top_k_rna": top_k_rna,
        "per_cell_k_rna": per_cell_k_rna,
        "top_k_met": top_k_met,
        "per_cell_k_met": per_cell_k_met,
        "lam_rna": lam_rna,
        "lam_met": lam_met,
        "tau_rna": tau_rna,
        "tau_met": tau_met,
        "do_clr_refine": True,
        "clr_lam2_rna": clr_lam2_rna,
        "clr_lam2_met": clr_lam2_met,
        "clr_iters": clr_iters,
        "fusion_temp": fusion_temp,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "use_unknown": use_unknown,
        "unknown_weight": unknown_weight,
        "scale_kind_rna": scale_kind_rna,
        "scale_kind_met": scale_kind_met,
    }


def sample_config_local(rng, center, sigma_log=0.5):
    """
    Local random search around `center`:
      - positive params: log10-normal jitter with std sigma_log
      - discrete params: +/- 1 step in choice list
      - booleans: rare flip
    """
    # Discrete grids (same as broad)
    TOPK_RNA = [1200, 1500, 2000, 2500, 3200, 4000, 5000, 6500, 8000]
    PER_RNA  = [150, 250, 400, 600, 800, 1000, 1200, 1500]
    TOPK_MET = [2000, 3000, 4000, 6000, 8000, 10000, 12000]
    PER_MET  = [200, 300, 500, 800, 1200, 1600, 2000]
    ITERS    = [20, 30, 40, 60, 80, 120]
    FTEMP    = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    AMIN     = [0.05, 0.10, 0.15, 0.20, 0.30, 0.35]
    AMAX     = [0.65, 0.75, 0.80, 0.90, 0.95]
    UWEIGHT  = [0.20, 0.30, 0.35, 0.45, 0.60]
    SKIND    = ["mean", "median"]

    cfg = dict(center)

    # Discrete jitters
    cfg["top_k_rna"] = jitter_int_choice(rng, int(cfg["top_k_rna"]), TOPK_RNA, max_step=1)
    cfg["per_cell_k_rna"] = jitter_int_choice(rng, int(cfg["per_cell_k_rna"]), PER_RNA, max_step=1)

    cfg["top_k_met"] = jitter_int_choice(rng, int(cfg["top_k_met"]), TOPK_MET, max_step=1)
    cfg["per_cell_k_met"] = jitter_int_choice(rng, int(cfg["per_cell_k_met"]), PER_MET, max_step=1)

    cfg["clr_iters"] = jitter_int_choice(rng, int(cfg["clr_iters"]), ITERS, max_step=1)

    # Positive (log-jitter)
    def log_jitter(x, lo, hi):
        lx = math.log10(max(x, lo))
        lx = lx + rng.normal(0.0, sigma_log)
        return float(10 ** clamp(lx, math.log10(lo), math.log10(hi)))

    cfg["lam_rna"] = log_jitter(float(cfg["lam_rna"]), 1e-8, 1e-1)
    cfg["lam_met"] = log_jitter(float(cfg["lam_met"]), 1e-8, 1e-1)

    cfg["tau_rna"] = log_jitter(float(cfg["tau_rna"]), 1e-9, 1e-2)
    cfg["tau_met"] = log_jitter(float(cfg["tau_met"]), 1e-9, 1e-2)

    cfg["clr_lam2_rna"] = log_jitter(float(cfg["clr_lam2_rna"]), 1e-8, 1e-1)
    cfg["clr_lam2_met"] = log_jitter(float(cfg["clr_lam2_met"]), 1e-8, 1e-1)

    # Fusion + clamps
    cfg["fusion_temp"] = float(rng.choice(FTEMP)) if rng.random() < 0.4 else float(cfg["fusion_temp"])
    cfg["alpha_min"] = float(rng.choice(AMIN)) if rng.random() < 0.3 else float(cfg["alpha_min"])
    cfg["alpha_max"] = float(rng.choice(AMAX)) if rng.random() < 0.3 else float(cfg["alpha_max"])
    if cfg["alpha_min"] >= cfg["alpha_max"]:
        cfg["alpha_min"], cfg["alpha_max"] = 0.2, 0.8

    # Unknown block
    if rng.random() < 0.10:
        cfg["use_unknown"] = not bool(cfg["use_unknown"])
    cfg["unknown_weight"] = float(rng.choice(UWEIGHT)) if rng.random() < 0.4 else float(cfg["unknown_weight"])

    # Scale kind
    cfg["scale_kind_rna"] = str(rng.choice(SKIND)) if rng.random() < 0.15 else str(cfg["scale_kind_rna"])
    cfg["scale_kind_met"] = str(rng.choice(SKIND)) if rng.random() < 0.15 else str(cfg["scale_kind_met"])

    # Always on in your pipeline
    cfg["do_clr_refine"] = True

    return cfg


def sample_config(rng, best_center, explore_prob=0.20, sigma_log=0.5):
    # With some probability explore broadly, otherwise exploit locally around current best
    if rng.random() < explore_prob or best_center is None:
        return sample_config_broad(rng)
    return sample_config_local(rng, best_center, sigma_log=sigma_log)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_config(datasets, ref_bulkRNA, ref_met, cfg, verbose=False):
    rows = []

    for ds in datasets:
        dataset_key = ds["dataset_key"]
        mixes = ds["mixes"]
        truth = ds["truth"]

        pred = program(
            mixes["mix_rna"],
            ref_bulkRNA,
            mix_met=mixes.get("mix_met", None),
            ref_met=ref_met,
            **cfg,
        )
        pred = clip_and_normalize(pred, eps=EPS)

        if truth is None:
            continue

        truth_aligned, pred_aligned = align_truth_pred(truth, pred, dataset_key=dataset_key)
        truth_aligned = clip_and_normalize(truth_aligned, eps=EPS)

        y_true = truth_aligned.T.to_numpy(dtype=float)
        y_pred = pred_aligned.T.to_numpy(dtype=float)

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

        if verbose:
            print(f"  [{dataset_key}] gscore={gscore:.6f} aitch={m_aitch:.3f} mae={m_mae:.3f} rmse={m_rmse:.3f}")

    if not rows:
        return pd.DataFrame([]), float("nan")

    summary = pd.DataFrame(rows)
    w = summary["n_samples"].to_numpy(dtype=float)
    overall_global = float(np.average(summary["global_score"], weights=w))
    return summary, overall_global


def load_all_datasets(project_path):
    data_dir = os.path.join(project_path, "data")
    gt_dir = os.path.join(project_path, "ground_truth")

    ref_path = os.path.join(data_dir, "reference_pdac.h5")
    ref = dp.read_hdf5(ref_path)
    ref_bulkRNA = ref["ref_bulkRNA"]
    ref_met = ref.get("ref_met", None)

    mix_files = [f for f in os.listdir(data_dir) if f.startswith("mixes_") and f.endswith(".h5")]
    mix_files.sort()

    datasets = []
    for mf in mix_files:
        dataset_key = mf.replace("mixes_", "").replace(".h5", "")
        mixes = dp.read_hdf5(os.path.join(data_dir, mf))

        truth = None
        if os.path.exists(gt_dir):
            truth = load_ground_truth(gt_dir, dataset_key)

        datasets.append({
            "dataset_key": dataset_key,
            "mixes": mixes,
            "truth": truth,
        })

    return datasets, ref_bulkRNA, ref_met


def normalize_project_path(p):
    p = os.path.abspath(p)
    if p.endswith(".h5"):
        p = os.path.dirname(os.path.dirname(p))
    return p


def print_summary_table(summary_df, overall_global, header):
    print(f"\n{header}")
    if summary_df is None or summary_df.empty:
        print("  (no datasets with ground truth found)")
        return

    cols_show = [
        "dataset", "n_samples",
        "global_score",
        "corr_block", "err_block", "dist_block",
        "aitchison_mean", "mae_mean", "rmse",
        "pearson_total", "pearson_cols", "pearson_rows",
        "spearman_total", "spearman_cols", "spearman_rows",
    ]
    summary_sorted = summary_df.sort_values("global_score", ascending=False)
    print(summary_sorted[cols_show].to_string(index=False))
    print(f"\nFINAL GLOBAL SCORE (weighted by n_samples): {overall_global:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-path", type=str, default=os.getcwd(), help="Path to starting_kit (default: cwd)")
    ap.add_argument("--trials", type=int, default=50, help="Number of random trials")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--show-top", type=int, default=5, help="Show top-k configs at end")
    ap.add_argument("--verbose", action="store_true", help="Print per-dataset gscore during trials")
    ap.add_argument("--save", type=str, default="tuning_results.jsonl", help="Write JSONL log to this file")

    # --- NEW (minimal): control exploit/explore behavior ---
    ap.add_argument("--explore-prob", type=float, default=0.20,
                    help="Probability of broad sampling (else local around current best)")
    ap.add_argument("--sigma-log", type=float, default=0.5,
                    help="Stddev of log10-jitter for positive params in local search")
    args = ap.parse_args()

    project_path = normalize_project_path(args.project_path)

    print("[LOAD] Loading datasets + references once...")
    datasets, ref_bulkRNA, ref_met = load_all_datasets(project_path)

    gt_datasets = [d for d in datasets if d["truth"] is not None]
    print(f"[LOAD] Found {len(datasets)} mixes files; {len(gt_datasets)} have ground truth.")

    rng = np.random.default_rng(args.seed)

    best_score = -1.0
    best_cfg = None
    best_summary = None

    topk = []

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("")

    # Start search centered on the known best config (then adapt center as you improve)
    current_center = dict(BEST_CFG_CENTER)

    for t in range(args.trials):
        cfg = sample_config(rng, current_center, explore_prob=args.explore_prob, sigma_log=args.sigma_log)
        summary_df, overall = evaluate_config(gt_datasets, ref_bulkRNA, ref_met, cfg, verbose=args.verbose)

        record = {"trial": t, "overall_global": overall, "config": cfg}
        if args.save:
            with open(args.save, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        print(f"trial {t:03d}/{args.trials} FINAL_GLOBAL={overall:.6f}")

        topk.append((overall, cfg, summary_df))
        topk.sort(key=lambda x: x[0], reverse=True)
        topk = topk[: max(1, args.show_top)]

        if overall > best_score:
            best_score = overall
            best_cfg = cfg
            best_summary = summary_df
            current_center = dict(best_cfg)  # <-- exploit around NEW best
            print("  NEW BEST ✅")
            print("  config:", json.dumps(best_cfg, indent=2))

    print("\n====================")
    print("BEST CONFIG FOUND")
    print("====================")
    print(json.dumps(best_cfg, indent=2))

    print_summary_table(best_summary, best_score, header="\n=== Best per-dataset summary ===")

    if args.show_top and len(topk) > 1:
        print("\n====================")
        print(f"TOP {len(topk)} CONFIGS")
        print("====================")
        for rank, (sc, cfg, _) in enumerate(topk, start=1):
            print(f"\n#{rank} FINAL_GLOBAL={sc:.6f}")
            print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
