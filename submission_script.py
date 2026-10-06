##################################################################################################
### PLEASE only edit the program function between YOUR CODE BEGINS/ENDS HERE                   ###
##################################################################################################


########################################################
### Package dependencies /!\ DO NOT CHANGE THIS PART ###
########################################################
import subprocess
import sys
import importlib

def program(mix=None, ref=None, **kwargs):
    ##
    ## YOUR CODE BEGINS HERE
    ##
    import numpy as np
    import pandas as pd
    from scipy.optimize import nnls

    EPS = 1e-6

    mix_met = kwargs.get("mix_met")
    ref_met = kwargs.get("ref_met")

    def clip_and_normalize_cells_x_samples(df, eps=EPS):
        X = df.to_numpy(dtype=float)
        X = np.clip(X, eps, 1.0 - eps)
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        X = X / col_sums
        X = np.clip(X, eps, 1.0 - eps)
        X = X / X.sum(axis=0, keepdims=True)
        return pd.DataFrame(X, index=df.index, columns=df.columns)

    def project_simplex(v, eps=EPS):
        v = np.asarray(v, dtype=float)
        v = np.maximum(v, 0.0)
        if v.sum() <= 0:
            v = np.ones_like(v) / len(v)

        # Keep a floor so CLR/log code does not run into zeros later
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]

        if len(rho) == 0:
            w = np.ones_like(v) / len(v)
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1.0) / (rho + 1)
            w = np.maximum(v - theta, 0.0)

        w = np.clip(w, eps, 1.0)
        w = w / w.sum()
        w = np.clip(w, eps, 1.0 - eps)
        w = w / w.sum()
        return w

    def clr_vec(x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, eps, None)
        lx = np.log(x)
        return lx - lx.mean()

    def clr_residual(A, w, b, eps=1e-12):
        m = A @ w
        r = clr_vec(m, eps=eps) - clr_vec(b, eps=eps)
        return float(np.mean(r * r))

    def scale_by_ref(mix_df, ref_df, eps=1e-8, kind="mean"):
        # I scale only here, the centering is separate and only used as a backup
        if kind == "median":
            scale = ref_df.median(axis=1).replace(0, eps)
        else:
            scale = ref_df.mean(axis=1).replace(0, eps)

        mix_s = mix_df.div(scale, axis=0)
        ref_s = ref_df.div(scale, axis=0)
        return mix_s, ref_s

    def center_and_shift_positive(mix_s, ref_s, eps=1e-8, shift=1e-3):
        mu = ref_s.mean(axis=1)
        mix_c = mix_s.sub(mu, axis=0)
        ref_c = ref_s.sub(mu, axis=0)

        feat_min = pd.concat([mix_c, ref_c], axis=1).min(axis=1)
        mix_p = mix_c.sub(feat_min, axis=0) + shift
        ref_p = ref_c.sub(feat_min, axis=0) + shift

        mix_p = mix_p.clip(lower=eps)
        ref_p = ref_p.clip(lower=eps)
        return mix_p, ref_p

    def select_marker_features(ref_df, top_k=4000, per_cell_k=800, eps=1e-8, modality="rna"):
        cells = list(ref_df.columns)
        markers = set()

        for c in cells:
            others = ref_df.drop(columns=[c]).mean(axis=1)

            if modality == "met":
                # For beta-like values, the ratios get noisy fast so we use absolute differences
                score = (ref_df[c] - others).abs()
            else:
                score = (ref_df[c] + eps) / (others + eps)

            markers |= set(score.sort_values(ascending=False).head(per_cell_k).index)

        markers = list(markers)
        if top_k is not None and top_k > 0 and len(markers) > top_k:
            v = ref_df.loc[markers].var(axis=1)
            markers = list(v.sort_values(ascending=False).head(top_k).index)

        return markers

    def ridge_nnls(A, b, lam=1e-3):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if lam is None or lam <= 0:
            w, _ = nnls(A, b)
            return w

        n_cells = A.shape[1]
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n_cells)])
        b_aug = np.concatenate([b, np.zeros(n_cells)])
        w, _ = nnls(A_aug, b_aug)
        return w

    def robust_ridge_nnls(A, b, lam=1e-3, iters=3, c=2.5, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        b = np.clip(b, eps, None)

        w = ridge_nnls(A, b, lam=lam)
        w = np.maximum(w, 0.0)

        for _ in range(max(0, int(iters))):
            m = np.clip(A @ w, eps, None)
            r = np.log(m) - np.log(b)

            weights = 1.0 / (1.0 + (r / float(c)) ** 2)
            weights = np.clip(weights, 1e-4, 1.0)

            sw = np.sqrt(weights)
            Aw = A * sw[:, None]
            bw = b * sw

            w = ridge_nnls(Aw, bw, lam=lam)
            w = np.maximum(w, 0.0)

        return w

    def refine_w_clr(A, b, w0, lam2=1e-3, iters=60, step=0.5, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        w = project_simplex(w0)
        y = clr_vec(b, eps=eps)

        def loss(w):
            m = np.clip(A @ w, eps, None)
            r = clr_vec(m, eps=eps) - y
            return float(np.mean(r * r) + lam2 * np.sum(w * w))

        cur = loss(w)

        for _ in range(iters):
            m = np.clip(A @ w, eps, None)
            r = clr_vec(m, eps=eps) - y
            r_centered = r - r.mean()

            grad_m = (2.0 * r_centered) / m
            grad_m *= 1.0 / len(m)
            grad_w = A.T @ grad_m + 2.0 * lam2 * w

            eta = step
            improved = False
            for _ in range(10):
                w_new = project_simplex(w - eta * grad_w)
                new = loss(w_new)

                if new <= cur:
                    w, cur = w_new, new
                    improved = True
                    break
                eta *= 0.5

            if not improved:
                break

        return w

    def refine_w_clr_robust(A, b, w0, lam2=1e-3, outer=3, inner=25, step=0.5, c=2.0, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        w = project_simplex(w0)
        y = clr_vec(b, eps=eps)
        wi = np.ones(A.shape[0], dtype=float)

        def wloss(w, wi):
            m = np.clip(A @ w, eps, None)
            r = clr_vec(m, eps=eps) - y
            return float((wi * (r * r)).mean() + lam2 * np.sum(w * w))

        cur = wloss(w, wi)

        for _ in range(max(1, int(outer))):
            m = np.clip(A @ w, eps, None)
            r = clr_vec(m, eps=eps) - y
            wi = 1.0 / (1.0 + (r / float(c)) ** 2)
            wi = np.clip(wi, 1e-4, 1.0)

            for _ in range(max(1, int(inner))):
                m = np.clip(A @ w, eps, None)
                r = clr_vec(m, eps=eps) - y

                r_mean_w = float(np.sum(wi * r) / (np.sum(wi) + 1e-12))
                r_wc = r - r_mean_w

                grad_m = (2.0 * wi * r_wc) / m
                grad_m *= 1.0 / len(m)
                grad_w = A.T @ grad_m + 2.0 * lam2 * w

                eta = step
                improved = False
                for _ in range(10):
                    w_new = project_simplex(w - eta * grad_w)
                    new = wloss(w_new, wi)

                    if new <= cur:
                        w, cur = w_new, new
                        improved = True
                        break
                    eta *= 0.5

                if not improved:
                    break

        return w

    def _rna_needs_log1p(mix_df, ref_df):
        mx = float(np.nanmax(mix_df.to_numpy()))
        rx = float(np.nanmax(ref_df.to_numpy()))

        mvals = mix_df.to_numpy().ravel()
        rvals = ref_df.to_numpy().ravel()
        mvals = mvals[np.isfinite(mvals)]
        rvals = rvals[np.isfinite(rvals)]

        mvals = mvals[: min(2000, len(mvals))]
        rvals = rvals[: min(2000, len(rvals))]

        intish = False
        if len(mvals) > 0 and len(rvals) > 0:
            intish = (
                np.mean(np.abs(mvals - np.round(mvals)) < 1e-6) > 0.9
                and np.mean(np.abs(rvals - np.round(rvals)) < 1e-6) > 0.9
            )

        return (mx > 50.0) or (rx > 50.0) or intish

    def add_unknown_signature(A_df, weight=0.35):
        # Used during fitting only, not in the final output
        avg = A_df.mean(axis=1)
        med = A_df.median(axis=1)
        unk = (0.7 * avg + 0.3 * med) * weight

        out = A_df.copy()
        out["__unknown__"] = unk
        return out

    def deconvolve_one_modality(
        mix_df,
        ref_df,
        top_k=4000,
        per_cell_k=800,
        lam=1e-3,
        tau_base=1e-5,
        modality="rna",
        scale_kind="mean",
        do_clr_refine=True,
        clr_lam2=1e-3,
        clr_iters=60,
    ):
        idx = mix_df.index.intersection(ref_df.index)
        mix_f = mix_df.loc[idx, :]
        ref_f = ref_df.loc[idx, :]

        if modality == "rna":
            if _rna_needs_log1p(mix_f, ref_f):
                # To avoid over-transforming already normalized data, we check if log1p seems needed based on the value distributions
                mix_f = np.log1p(mix_f)
                ref_f = np.log1p(ref_f)

        # Keep original behavior here
        feats = select_marker_features(ref_f, top_k=top_k, per_cell_k=per_cell_k, eps=1e-8)
        mix_f = mix_f.loc[feats, :]
        ref_f = ref_f.loc[feats, :]

        mix_s, ref_s = scale_by_ref(mix_f, ref_f, eps=1e-8, kind=scale_kind)

        auto_center = bool(kwargs.get("auto_center", True))
        center_shift = float(kwargs.get("center_shift", 1e-3))

        def probe_mismatch(mix_s, ref_s, lam_probe):
            A0 = ref_s.to_numpy(dtype=float)
            cols = list(mix_s.columns)
            probe_cols = cols[: min(5, len(cols))]

            clrs = []
            rels = []

            for c in probe_cols:
                b0 = mix_s[c].to_numpy(dtype=float)
                w0 = ridge_nnls(A0, b0, lam=lam_probe)
                w0 = project_simplex(w0)

                clrs.append(clr_residual(A0, w0, b0, eps=1e-12))
                m0 = A0 @ w0
                rels.append(float(np.linalg.norm(m0 - b0) / (np.linalg.norm(b0) + 1e-12)))

            return float(np.median(clrs)), float(np.median(rels))

        do_center = False
        if auto_center:
            # Centering can help shift, but it can also hurt clean samples
            clr_med, rel_med = probe_mismatch(mix_s, ref_s, lam_probe=max(lam, 1e-8))
            clr_thr = float(kwargs.get("auto_center_clr_thr", 0.55))
            rel_thr = float(kwargs.get("auto_center_rel_thr", 0.28))
            do_center = (clr_med > clr_thr) and (rel_med > rel_thr)

        if do_center:
            mix_s, ref_s = center_and_shift_positive(mix_s, ref_s, eps=1e-8, shift=center_shift)

        use_unknown = kwargs.get("use_unknown", True)
        unknown_weight = float(kwargs.get("unknown_weight", 0.6))
        ref_fit = add_unknown_signature(ref_s, weight=unknown_weight) if use_unknown else ref_s

        A = ref_fit.to_numpy(dtype=float)
        fit_cells = list(ref_fit.columns)
        sample_ids = list(mix_s.columns)

        W = np.zeros((len(fit_cells), len(sample_ids)), dtype=float)
        clr_resids = np.zeros(len(sample_ids), dtype=float)

        robust_enable = bool(kwargs.get("robust_enable", True))
        robust_threshold_clr = float(kwargs.get("robust_threshold_clr", 0.40))
        robust_threshold_relerr = float(kwargs.get("robust_threshold_relerr", 0.25))
        robust_unknown_gate = float(kwargs.get("robust_unknown_gate", 0.18))

        irls_iters = int(kwargs.get("robust_irls_iters", 3))
        irls_c = float(kwargs.get("robust_irls_c", 2.5))

        robust_outer = int(kwargs.get("robust_clr_outer", 3))
        robust_inner = int(kwargs.get("robust_clr_inner", 20))
        robust_c_clr = float(kwargs.get("robust_clr_c", 2.0))
        robust_step = float(kwargs.get("robust_step", 0.5))

        trim_enable = bool(kwargs.get("trim_enable", True))
        trim_frac = float(kwargs.get("trim_frac", 0.15))
        trim_min_keep = int(kwargs.get("trim_min_keep", 2500))
        trim_trigger = float(kwargs.get("trim_trigger", 0.45))

        for j, s in enumerate(sample_ids):
            b = mix_s[s].to_numpy(dtype=float)

            w0 = ridge_nnls(A, b, lam=lam)
            w0 = np.maximum(w0, 0.0)
            w0 = w0 / (w0.sum() + 1e-12)

            w0s = project_simplex(w0)
            base_clr = clr_residual(A, w0s, b, eps=1e-12)
            m0 = A @ w0s
            relerr = float(np.linalg.norm(m0 - b) / (np.linalg.norm(b) + 1e-12))

            unknown_frac = 0.0
            if "__unknown__" in fit_cells:
                unknown_frac = float(w0s[fit_cells.index("__unknown__")])

            shifted = False
            if robust_enable:
                shifted = (
                    ((base_clr > robust_threshold_clr) and (relerr > robust_threshold_relerr))
                    or (unknown_frac > robust_unknown_gate)
                )

            if shifted:
                # Robust path is slower, so only use it when the base fit already looks off
                w0 = robust_ridge_nnls(A, b, lam=lam, iters=irls_iters, c=irls_c, eps=1e-12)
                w0 = np.maximum(w0, 0.0)
                w0 = w0 / (w0.sum() + 1e-12)

            # Light smoothing keeps refinement away from the simplex boundary
            p0 = w0 / (w0.sum() + 1e-12)
            entropy = -np.sum(p0 * np.log(p0 + 1e-12))
            tau_eff = tau_base * np.exp(-entropy)
            w0 = w0 + tau_eff
            w0 = w0 / (w0.sum() + 1e-12)

            if do_clr_refine:
                if shifted:
                    w = refine_w_clr_robust(
                        A, b, w0,
                        lam2=clr_lam2,
                        outer=robust_outer,
                        inner=robust_inner,
                        step=robust_step,
                        c=robust_c_clr,
                        eps=1e-12,
                    )
                else:
                    w = refine_w_clr(A, b, w0, lam2=clr_lam2, iters=clr_iters, step=0.5, eps=1e-12)
            else:
                w = project_simplex(w0)

            if trim_enable and shifted:
                # One retry after dropping the worst residual features
                cur_clr = clr_residual(A, w, b, eps=1e-12)
                if cur_clr > trim_trigger:
                    m = np.clip(A @ w, 1e-12, None)
                    r = clr_vec(m, eps=1e-12) - clr_vec(b, eps=1e-12)
                    absr = np.abs(r)

                    n_feat = len(absr)
                    keep_n = int(max(trim_min_keep, (1.0 - trim_frac) * n_feat))
                    keep_n = min(keep_n, n_feat)
                    keep_idx = np.argsort(absr)[:keep_n]

                    A_t = A[keep_idx, :]
                    b_t = b[keep_idx]

                    w_t = robust_ridge_nnls(A_t, b_t, lam=lam, iters=irls_iters, c=irls_c, eps=1e-12)
                    w_t = np.maximum(w_t, 0.0)
                    w_t = w_t / (w_t.sum() + 1e-12)

                    pT = w_t / (w_t.sum() + 1e-12)
                    entT = -np.sum(pT * np.log(pT + 1e-12))
                    w_t = w_t + tau_base * np.exp(-entT)
                    w_t = w_t / (w_t.sum() + 1e-12)

                    w_t = refine_w_clr_robust(
                        A_t, b_t, w_t,
                        lam2=clr_lam2,
                        outer=max(1, robust_outer),
                        inner=max(10, robust_inner),
                        step=robust_step,
                        c=robust_c_clr,
                        eps=1e-12,
                    )

                    new_clr_t = clr_residual(A_t, w_t, b_t, eps=1e-12)
                    if new_clr_t < cur_clr:
                        w = w_t

            clr_resids[j] = clr_residual(A, w, b, eps=1e-12)
            W[:, j] = w

        out_fit = pd.DataFrame(W, index=fit_cells, columns=sample_ids)
        out_fit = clip_and_normalize_cells_x_samples(out_fit, eps=EPS)

        # Final output has to be only the expected cell types
        if use_unknown and "__unknown__" in out_fit.index:
            out = out_fit.drop(index="__unknown__")
            out = clip_and_normalize_cells_x_samples(out, eps=EPS)
        else:
            out = out_fit

        return out, pd.Series(clr_resids, index=sample_ids)

    if mix is None or ref is None:
        raise ValueError("mix and ref must be provided (RNA modality).")

    top_k_rna = kwargs.get("top_k_rna", 3200)
    per_cell_k_r = kwargs.get("per_cell_k_rna", 400)
    lam_rna = kwargs.get("lam_rna", 1.0861693402420861e-05)
    tau_rna = kwargs.get("tau_rna", 0.00047545192734529197)
    scale_kind_r = kwargs.get("scale_kind_rna", "mean")

    do_clr_refine = kwargs.get("do_clr_refine", True)
    clr_lam2_rna = kwargs.get("clr_lam2_rna", 0.001672544697416985)
    clr_iters = kwargs.get("clr_iters", 20)

    prop_rna, res_rna = deconvolve_one_modality(
        mix,
        ref,
        top_k=top_k_rna,
        per_cell_k=per_cell_k_r,
        lam=lam_rna,
        tau_base=tau_rna,
        modality="rna",
        scale_kind=scale_kind_r,
        do_clr_refine=do_clr_refine,
        clr_lam2=clr_lam2_rna,
        clr_iters=clr_iters,
    )

    if isinstance(mix_met, pd.DataFrame) and isinstance(ref_met, pd.DataFrame):
        top_k_met = kwargs.get("top_k_met", 2000)
        per_cell_k_m = kwargs.get("per_cell_k_met", 2000)
        lam_met = kwargs.get("lam_met", 0.0021459430390383017)
        tau_met = kwargs.get("tau_met", 1e-09)
        scale_kind_m = kwargs.get("scale_kind_met", "median")
        clr_lam2_met = kwargs.get("clr_lam2_met", 3.407091789101604e-08)

        prop_met, res_met = deconvolve_one_modality(
            mix_met,
            ref_met,
            top_k=top_k_met,
            per_cell_k=per_cell_k_m,
            lam=lam_met,
            tau_base=tau_met,
            modality="met",
            scale_kind=scale_kind_m,
            do_clr_refine=do_clr_refine,
            clr_lam2=clr_lam2_met,
            clr_iters=clr_iters,
        )

        common_cells = prop_rna.index.intersection(prop_met.index)
        common_samp = prop_rna.columns.intersection(prop_met.columns)

        prop_rna = prop_rna.loc[common_cells, common_samp]
        prop_met = prop_met.loc[common_cells, common_samp]
        res_rna = res_rna.loc[common_samp]
        res_met = res_met.loc[common_samp]

        # Lower residual gets more weight
        t = float(kwargs.get("fusion_temp", 0.3))
        wr = np.exp(-res_rna / t)
        wm = np.exp(-res_met / t)
        alpha_s = wr / (wr + wm + 1e-12)

        meltdown_on = bool(kwargs.get("fusion_meltdown_enable", True))
        meltdown_thr = float(kwargs.get("fusion_meltdown_rna_clr_thr", 5.0))
        meltdown_alpha = float(kwargs.get("fusion_meltdown_alpha", 0.01))

        if meltdown_on:
            # If RNA is obviously broken for a sample, let MET take over
            bad_rna = res_rna > meltdown_thr
            alpha_s = alpha_s.where(~bad_rna, meltdown_alpha)

        a_min = float(kwargs.get("alpha_min", 0.05))
        a_max = float(kwargs.get("alpha_max", 0.65))

        if meltdown_on:
            alpha_s = alpha_s.clip(lower=min(a_min, meltdown_alpha), upper=a_max)
        else:
            alpha_s = alpha_s.clip(lower=a_min, upper=a_max)

        prop = prop_rna.mul(alpha_s, axis=1) + prop_met.mul((1.0 - alpha_s), axis=1)
        prop = clip_and_normalize_cells_x_samples(prop, eps=EPS)
    else:
        prop = prop_rna

    prop = prop.reindex(columns=mix.columns)
    return prop
    ##
    ## YOUR CODE ENDS HERE
    ##
    
# Install and import each package
def install_and_import_packages(required_packages):
  for package in required_packages:
      try:
          globals()[package] = importlib.import_module(package)
      except ImportError:
          print('impossible to import, installing packages',package)
          package_to_install = 'scikit-learn' if package == 'sklearn' else package
          subprocess.check_call([sys.executable, "-m", "pip", "install", package_to_install])
          globals()[package] = importlib.import_module(package)
          

# Install and import each package
def install_and_import_packages(required_packages):
    def try_pip_install(package_name):
      """Try pip install; detect externally-managed-environment error."""
      try:
          subprocess.check_call(
              [sys.executable, "-m", "pip", "install", package_name]
          )
          return True
      except subprocess.CalledProcessError as e:
          if "externally-managed-environment" in str(e):
              return False  # pip blocked by PEP 668
          raise  # real error unrelated to PEP 668
  

    def try_conda_install(package_name):
        """Attempt conda install."""
        try:
            subprocess.check_call(["conda", "install", "-y", package_name])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    for package in required_packages:
        try:
            globals()[package] = importlib.import_module(package)
        except ImportError:
            print('impossible to import, installing packages',package)
            package_to_install = 'scikit-learn' if package == 'sklearn' else package
            pip_ok = try_pip_install(package_to_install)

            if pip_ok:
              globals()[package] = importlib.import_module(package)
              continue

            # Pip failed due to externally-managed environment (Debian, Conda etc.)
            print("⛔ pip installation blocked by externally-managed environment.")
            print(f"➡️ Trying conda install: {package_to_install}")

            conda_ok = try_conda_install(package_to_install)

            if conda_ok:
                globals()[package] = importlib.import_module(package)
                continue
            print(f"Unable to install package {package_to_install} automatically.")

def validate_pred(pred, nb_samples=None, nb_cells=None, col_names=None):
    error_status = 0  # 0 means no errors, 1 means "Fatal errors" and 2 means "Warning"
    error_informations = ''

    # Ensure that all sum of cells proportion approximately equal 1
    if not numpy.allclose(numpy.sum(pred, axis=0), 1):
        msg = "The prediction matrix does not respect the laws of proportions: the sum of each column should be equal to 1\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction has the correct names
    if not set(col_names) == set(pred.index):
        msg = f"The row names in the prediction matrix should match: {col_names}\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction returns the correct number of samples and number of cells
    if pred.shape != (nb_cells, nb_samples):
        msg = f'The prediction matrix has the dimension: {pred.shape} whereas the dimension: {(nb_cells, nb_samples)} is expected\n'
        error_informations += msg
        error_status = 1

    if error_status == 1:
        # The error is blocking and should therefore stop the execution
        raise ValueError(error_informations)
    if error_status == 2:
        print("Warning:")
        print(error_informations)


##############################################################
### Generate a prediction file /!\ DO NOT CHANGE THIS PART ###
##############################################################

# List of required packages
required_packages = [
  "numpy",
  "pandas",
  "zipfile",
  "inspect",
  "h5py",
  "scipy",
  "sklearn"
]
install_and_import_packages(required_packages)

import os
import attachement.data_processing as dp


dir_name = "data"+os.sep

datasets_list = [filename for filename in os.listdir(dir_name) if filename.startswith("mixes")]

ref_file = os.path.join(dir_name, "reference_pdac.h5")
reference_data = dp.read_hdf5(ref_file)


predi_dic = {}
for dataset_name in datasets_list :

    file= os.path.join(dir_name,dataset_name)
    mixes_data = dp.read_hdf5(file)

    print(f"generating prediction for dataset: {dataset_name}")

    # mix_rna = extract_data_element(mixes_data,file, 'mix_rna') 
    # mix_met = extract_data_element(mixes_data,file, 'mix_met')
    cleaned_name=dataset_name.replace("mixes_", "").removesuffix(".h5")

    pred_prop = program(mixes_data["mix_rna"], reference_data["ref_bulkRNA"], mix_met=mixes_data["mix_met"], ref_met=reference_data["ref_met"]   )
    # validate_pred(pred_prop, nb_samples=mix_rna.shape[1], nb_cells=ref_bulkRNA.shape[1], col_names=ref_bulkRNA.columns)
    predi_dic[cleaned_name] = pred_prop

############################### 
### Code submission mode

# we generate a zip file with the 'program' source code

if not os.path.exists("submissions"):
    os.makedirs("submissions")

# we save the source code as a Python file named 'program.py':
with open(os.path.join("submissions", "program.py"), 'w') as f:
    f.write(inspect.getsource(program))

date_suffix = pandas.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")




# we create the associated zip file:
zip_program = os.path.join("submissions", f"program_{date_suffix}.zip")
with zipfile.ZipFile(zip_program, 'w') as zipf:
    zipf.write(os.path.join("submissions", "program.py"), arcname="program.py")


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
if os.path.exists("attachement"):
    with zipfile.ZipFile(zip_program, 'a', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('attachement/', zipf)


print(zip_program)

###############################


# Generate a zip file with the prediction
if not os.path.exists("submissions"):
    os.makedirs("submissions")

prediction_name = "prediction.h5"

dp.write_hdf5(os.path.join("submissions", prediction_name),predi_dic)



# Create the associated zip file:
zip_results = os.path.join("submissions", f"results_{date_suffix}.zip")
with zipfile.ZipFile(zip_results, 'w') as zipf:
    zipf.write(os.path.join("submissions", prediction_name), arcname=prediction_name)

print(zip_results)

