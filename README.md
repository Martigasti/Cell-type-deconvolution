# Multi-omics Cell-Type Deconvolution for Pancreatic Cancer

Estimate the **proportions of cell types** in bulk tumor samples of pancreatic ductal adenocarcinoma (PDAC) from two modalities, **bulk RNA-seq** and **DNA methylation**, using reference profiles of pure cell types.
Built for a deconvolution data challenge in the M1 Data Science "DataCamp" course (Université Paris-Saclay, 2025–2026).

Mathematically, each bulk sample `b` is modeled as a mixture `b ≈ A·w` of reference profiles `A`, where `w` must lie on the probability simplex (non-negative, sums to 1). Real data breaks this model in several ways: platform shifts, missing cell types, noisy features. Most of the work went into making the solver robust to those failures.

## Method (`submission_script.py`, function `program`)

The same pipeline runs per modality, and the two results are then fused:

1. **Marker selection**: keep the most cell-type-specific features (top-k global plus top-k per cell type). It uses expression ratios for RNA and absolute differences for methylation beta values.
2. **Normalization**: per-feature scaling by the reference, with automatic `log1p` detection for RNA. Centering is used **only when a probe fit detects a distribution shift**, because it hurts clean samples.
3. **Constrained solver**: ridge-regularized NNLS, followed by projection onto the simplex.
4. **Escalation for samples that fit badly** (decided per sample from the CLR residual and the relative error):
   - robust IRLS-NNLS that down-weights outlier features,
   - an extra "unknown" signature during fitting that absorbs cell types missing from the reference, dropped from the output,
   - trimming of the worst-residual features with one refit.
5. **Compositional refinement**: gradient refinement of `w` in CLR (Aitchison) space, the geometry the evaluation uses.
6. **RNA + methylation fusion**: a per-sample convex combination weighted by `exp(−residual / T)`. If the RNA fit collapses for a sample, methylation takes over.

The hyperparameters (feature counts, ridge strengths, fusion temperature, …) come from a 200-trial random search ([`tune_hyperparams.py`](tune_hyperparams.py), log in [`tuning_results.jsonl`](tuning_results.jsonl)). [`vivo_diagnostics.py`](vivo_diagnostics.py) prints per-dataset fit diagnostics (unknown-signature weight, CLR residual, relative error) that were used to understand the hardest dataset.

## Results (local evaluation, `test_score.py`)

The global score combines a correlation block (Pearson and Spearman), an error block (MAE and RMSE) and an Aitchison-distance block. Higher is better.

| Dataset | Samples | Global score |
|---|---|---|
| SDN4 | 30 | 0.815 |
| SDN5 | 30 | 0.735 |
| SDN6 | 30 | 0.718 |
| SDE5 | 30 | 0.704 |
| SDC5 | 30 | 0.668 |
| SDEL | 30 | 0.576 |
| SBN5 | 30 | 0.551 |
| VITR (in vitro) | 15 | 0.483 |
| VIVO (in vivo) | 24 | 0.389 |
| **Weighted overall** | 249 | **0.641** |

The random-search trials ranged from 0.567 to 0.640. Synthetic mixtures are recovered well (Pearson 0.84–0.95). The in-vivo samples remain the hard case: their profiles diverge strongly from the references, which is exactly what the robust and fusion paths are designed to mitigate.

## Usage

The challenge data is not included. Put the files in `data/` (`reference_pdac.h5`, `mixes_*.h5`) and the ground truth in `ground_truth/` (`groundtruth_*.h5`).

```bash
pip install numpy pandas scipy scikit-learn h5py
python submission_script.py     # writes predictions and the zipped submission to submissions/
python test_score.py            # local evaluation against ground truth
python tune_hyperparams.py --trials 50 --seed 42
```

`attachement/data_processing.py` is the challenge's HDF5 I/O helper, provided by the organizers.
