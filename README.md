# Sparse Lens Certification

Official code release for **“When Can We Trust the Sparse Lens? A Certification Framework for SAE Faithfulness”**, accepted to *Transactions of the Association for Computational Linguistics (TACL)*.

This repository contains the code and released intermediate outputs for the certification experiments in the paper. The framework asks whether a pretrained sparse autoencoder (SAE) reconstruction preserves enough of a frozen language model's predictive behavior to support a non-vacuous population-risk certificate for the original model.

## What the framework certifies

For a frozen language model \(M\), we insert a pretrained SAE at a chosen internal layer and continue the forward pass from the SAE reconstruction. The resulting patched model acts as an SAE-based proxy.

The main fixed-pool certificate, \(U_{\mathrm{SAE}}\), combines:

1. the empirical risk of the calibration-selected SAE proxy;
2. the reconstruction-loss gap between the original LM and the unrestricted SAE reconstruction; and
3. support mismatch between the calibration-derived feature pool and held-out inputs.

The optional exact-\(P\) extension, \(U_{\mathrm{SAE+Occam}}\), replaces the fixed-proxy risk concentration term with one that is uniform over all feature pools containing exactly \(P\) features. Its pool-identity complexity is

\[
C_P = \ln \binom{m}{P},
\]

where \(m\) is the SAE dictionary size and \(P\) is the calibration-pool size.

The exact-\(P\) term is **not required** for ordinary held-out certification of the fixed calibration-selected proxy; it is reported as an optional, more conservative extension.

---

## Repository layout

The directory names reflect the internal organization used while producing the experiments. The corresponding paper items are:

| Paper item | Code / released outputs |
|---|---|
| Figure 1 + Table 1: cross-model certification | `fig1_tab1/plot2_batched_topk_unified_gemma_gpt_llama_delta_exact_hoeffding_final.py` |
| Table 2: calibration-size sensitivity | `fig1_tab1/calibration_size_sensitivity_unified_updated.py` |
| Figure 2 + Tables 3–4: Llama layerwise certification and output fidelity | `fig2_tab2_tab3/` |
| Figure 3: horizon-conditioned base–proxy KL | `fig3/plot2_horizon_kl_unified_gemma_gpt_llama_layer_cached.py` |
| Figure 5: fixed-global feature-identity shuffling | `fig4/shuffled_features_unified_gemma_gpt_llama_global_permutation.py` |
| Figure 6: GPT-2 Small layerwise certificates | `fig5/plot2_batched_gpt2_layerwise_delta_exact_hoeffding_final_dual_plots.py` |
| Figure 7: Top-\(k\) sensitivity | per-\(k\) runs and outputs in `fig1_tab1/final_ops_{16,32,64,128}/` |
| Figure 8: synthetic-corruption certificate decomposition | `fig7/bound_decomposition_unified_gpt2_gemma_llama_updated.py` |
| Table 7: downstream-task preservation | `tab7/downstream_performance.py` |
| Table 8: frequency-thresholded pools | `frequency/llama_layer30_frequency_thresholded_pools.py` |

Precomputed CSV/PDF/PNG outputs from the reported runs are included in the corresponding subdirectories.




---

## Environment

The scripts use:

- Python
- PyTorch
- NumPy
- pandas
- Matplotlib
- seaborn
- Hugging Face `datasets`
- `tqdm`
- TransformerLens
- SAE Lens
- EleutherAI `sparsify`

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  torch \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  datasets \
  tqdm \
  transformer-lens \
  sae-lens \
  eai-sparsify
```

The released archive does not contain a pinned lockfile, so the exact original package versions are not specified.

The Llama experiments load:

```text
meta-llama/Meta-Llama-3-8B
EleutherAI/sae-llama-3-8b-32x
```

Make sure your Hugging Face account has access to the required gated checkpoints and authenticate before running, e.g.

```bash
hf auth login
```

A CUDA-capable GPU is strongly recommended. The scripts fall back to CPU where supported, but the Llama experiments are computationally expensive.

---

## Common experimental settings

Unless a script overrides them explicitly, the reported experiments use:

```text
Dataset                 allenai/c4, English train split
Sequence length         32 tokens
Calibration budget      2.24M tokens
Default Top-k           64
Smoothing alpha         0.5
Confidence delta        0.05
Calibration/evaluation  disjoint streams
```

The three primary model/SAE settings are:

| Model | SAE | Hook |
|---|---|---|
| GPT-2 Small | `gpt2-small-res-jb` | `blocks.6.hook_resid_pre` |
| Gemma-2B | `gemma-2b-res-jb` | `blocks.12.hook_resid_post` |
| Llama-3-8B | `EleutherAI/sae-llama-3-8b-32x` | `blocks.30.hook_resid_post` |

For GPT-2 Small and Gemma-2B, the code uses SAE Lens checkpoints. For Llama-3-8B, it uses EleutherAI's `sparsify` SAE.

The common evaluation-time Top-\(k\) rule is an experimental protocol; it is not assumed to be the native coding rule of every released SAE.

---

# Reproducing the experiments

Because several scripts use relative paths, **run each command from the directory that contains the script**.

## 1. Figure 1 and Table 1: cross-model certification

```bash
cd fig1_tab1
python plot2_batched_topk_unified_gemma_gpt_llama_delta_exact_hoeffding_final.py
```

The script evaluates GPT-2 Small, Gemma-2B, and Llama-3-8B and writes a per-model parameter CSV, the combined bound-results CSV, and the certificate plot under

```text
final_ops_<k>/
```

For the main-paper setting, use:

```python
topk = 64
```

near the top of the script.

**Important:** the checked-in driver currently has `topk = 128`, corresponding to a Top-\(k\) sensitivity run. Set it to `64` to reproduce the main Figure 1 / Table 1 setting.

For \(k=64\), the released outputs are:

```text
fig1_tab1/final_ops_64/GPT-2_Small_bound_params30.csv
fig1_tab1/final_ops_64/Gemma-2B_bound_params30.csv
fig1_tab1/final_ops_64/Llama-3_1-8B_bound_params30.csv
fig1_tab1/final_ops_64/bound_results_v2_unified30_k=64_final.csv
fig1_tab1/final_ops_64/concept_pool_bound_plot_unified_better30_k=64_final.{png,pdf}
```

The filename keeps `k=64` in its name even when the driver is run with another `topk`; the enclosing `final_ops_<k>/` directory records the actual setting.

---

## 2. Table 2: calibration-size sensitivity

Use the updated calibration-sensitivity script:

```bash
cd fig1_tab1

python calibration_size_sensitivity_unified_updated.py
```

Default calibration budgets are:

```text
280k, 560k, 1.12M, 2.24M tokens
```

All pools are nested prefixes of the same shuffled calibration stream and are evaluated on the same held-out stream beginning after the largest calibration budget.

The default output is:

```text
calibration_sensitivity/calibration_size_sensitivity_table.csv
```

Useful options include:

```bash
python calibration_size_sensitivity_unified_updated.py \
  --calibration_budgets 280000 560000 1120000 2240000 \
  --evaluation_tokens 320000 \
  --top_k 64 \
  --seq_len 32 \
  --alpha 0.5 \
  --delta 0.05 \
  --out_dir ./calibration_sensitivity
```

A subset of models can be selected with `--models`.

---

## 3. Figure 2 and Tables 3–4: Llama-3-8B layerwise analysis

The reported Llama layers are:

```text
4, 8, 12, 16, 20, 24, 28, 30
```

### 3.1 Generate the layerwise certificate CSVs

```bash
cd fig2_tab2_tab3
mkdir -p bounds final_ops runs

bash run_plot2_batched_llama_layers.sh
```

or run a single layer with:

```bash
python plot2_batched_llama_layered_delta_exact_hoeffding_final.py --layer 30
```

The per-layer outputs are written to:

```text
bounds/llama3_8b_layers_<LAYER>.csv
bounds/bound_results_layer_<LAYER>.csv
```

and an individual layer plot is written to:

```text
final_ops/concept_pool_bound_plot_llama3_8b_eleuther_layer_<LAYER>.png
```

### 3.2 Generate output-fidelity CSVs

```bash
bash run_output_fidelity_llama_layers.sh
```

or run one layer explicitly:

```bash
python output_fidelity_batched_llama_layered_delta_exact_hoeffding_final.py \
  --layer 30 \
  --top_k 64 \
  --seq_len 32 \
  --calibration_tokens 2240000 \
  --out_dir ./final_ops
```

This produces:

```text
final_ops/output_fidelity_layer_<LAYER>.csv
final_ops/output_fidelity_kl_layer_<LAYER>.png
final_ops/output_fidelity_top1_layer_<LAYER>.png
```

### 3.3 Build the combined Figure 2 and the final layerwise crossing summary

For the **final paper's 10k, 30k, 50k, ... descriptive \(N^\star\) grid**, use:

```bash
python plot_llama_depth_grid20k_final.py
```

This consumes:

```text
bounds/llama3_8b_layers_*.csv
final_ops/output_fidelity_layer_*.csv
```

and writes:

```text
final_ops/llama_bound_layers.{png,pdf}
final_ops/llama_output_fidelity_layers.{png,pdf}
```

The script reproduces the final-paper descriptive crossings:

```text
Layer 4   Never
Layer 8   Never
Layer 12  Never
Layer 16  Never
Layer 20  450k
Layer 24  150k
Layer 28  90k
Layer 30  10k
```


---


## 4. Figure 3: horizon-conditioned base–proxy KL

Run the horizon-conditioned analysis from the `fig3/` directory:

```bash
cd fig3
python plot2_horizon_kl_unified_gemma_gpt_llama_layer_cached.py
```

The script compares the base model and its SAE-reconstructed proxy after matched rollout horizons:

```text
0, 1, 2, 4, 8, full
```

where `full` means running the complete remaining model computation to the final next-token distribution. For finite horizons, the script compares base and proxy residual-stream states after the same number of downstream blocks and reports the mean next-token KL divergence in bits.

The default configuration uses:

```text
Evaluation target       32,000 tokens
Sequence length         32 tokens
Top-k                    64
GPT-2 Small layer       6
Gemma-2B layer           12
Llama-3-8B layers        4, 8, 12, 16, 20, 24, 28, 30
Horizons                 0, 1, 2, 4, 8, full
```

Token batches are materialized once per model and reused across patch layers. Because evaluation proceeds in complete batches, the realized token count can slightly exceed the nominal 32,000-token target (the released run cached 32,256 tokens).

Outputs:

```text
final_ops/horizon_conditioned_proxy_kl_results.csv
final_ops/horizon_conditioned_proxy_kl_plot.png
final_ops/horizon_conditioned_proxy_kl_plot.pdf
```

The CSV records, for each model, patch layer, and rollout horizon:

```text
Mean KL (bits)
Std KL (bits)
Num Batches
Tokens Cached
```

This experiment is used to distinguish **local SAE reconstruction fidelity** from **downstream amplification of reconstruction error**. In the Llama-3-8B layer sweep, shorter horizons probe local disagreement near the patch location, while the longer horizons show how that disagreement evolves through the remaining computation.

---

## 5. Figure 5: feature-identity shuffling

```bash
cd fig4
mkdir -p final_ops_final

python shuffled_features_unified_gemma_gpt_llama_global_permutation.py
```

The experiment samples **one fixed global bijection per model-SAE pair** and reuses it throughout the evaluation. This preserves the active coefficient values, local sparsity, calibration-pool size \(P\), and exact-\(P\) complexity while changing feature identities.

Outputs:

```text
final_ops_final/ablation_shuffled_histogram_unified30_final.png
final_ops_final/ablation_shuffled_histogram_unified30_final.pdf
```

---

## 6. Figure 6: GPT-2 Small layerwise certificates

```bash
cd fig5

python plot2_batched_gpt2_layerwise_delta_exact_hoeffding_final_dual_plots.py
```

The script evaluates GPT-2 Small at layers:

```text
0, 2, 4, 6, 8, 10
```

with held-out sequence checkpoints:

```text
10k, 30k, 50k, 70k, 140k
```

Outputs are written to:

```text
gpt2_layerwise_exact_final/
```

including:

```text
gpt2_layerwise_delta_exact_hoeffding_results.csv
gpt2_bounds_across_layers_u_sae.{png,pdf}
gpt2_bounds_across_layers_u_sae_occam.{png,pdf}
gpt2_layerwise_crossing_summary_both_bounds.csv
```

The two plots correspond to the fixed-pool \(U_{\mathrm{SAE}}\) certificate and the optional exact-\(P\) \(U_{\mathrm{SAE+Occam}}\) extension.

---

## 7. Figure 7: Top-\(k\) sensitivity

The cross-model driver in `fig1_tab1/` is also used for the Top-\(k\) sensitivity runs.

Set the top-level variable

```python
topk = ...
```

to each of:

```text
16, 32, 64, 128
```

and run:

```bash
python plot2_batched_topk_unified_gemma_gpt_llama_delta_exact_hoeffding_final.py
```

The release already contains the per-\(k\) result directories:

```text
fig1_tab1/final_ops_16/
fig1_tab1/final_ops_32/
fig1_tab1/final_ops_64/
fig1_tab1/final_ops_128/
```

These contain the per-\(k\) certificate curves and CSVs used for the sensitivity analysis.

---

## 8. Figure 8: synthetic input corruption

```bash
cd fig7

python bound_decomposition_unified_gpt2_gemma_llama_updated.py
```

The code first builds a clean C4 calibration pool and then evaluates the same fixed pool under:

```text
IID          clean English
Slight-OOD   15% random token replacement
Far-OOD      random-token sequences
```

The default script uses 2.24M calibration tokens and 2.24M evaluation tokens with Top-\(k=64\).

Outputs:

```text
final_ops_corruption_updated/certificate_decomposition_synthetic_corruption_updated.csv
final_ops_corruption_updated/certificate_decomposition_synthetic_corruption_updated.png
final_ops_corruption_updated/certificate_decomposition_synthetic_corruption_updated.pdf
```

---


## 9. Table 7: downstream-task preservation

The downstream experiment compares three predictors:

```text
M        original frozen language model
S∘M      unrestricted SAE-reconstructed proxy
h_G      calibration-pool-restricted SAE proxy
```

Run:

```bash
cd tab6
python downstream_performance.py
```

The directory is named `tab6/` in the released source for historical reasons; this script reproduces **Table 7** in the final paper.

The default configuration uses:

```text
Top-k                     64
Sequence length           32 tokens
Calibration budget        2.24M C4 tokens
Examples per dataset      1,000
GPT-2 Small layer         6
Gemma-2B layer            12
Llama-3-8B layers         4, 8, 12, 16, 20, 24, 28, 30
```

The script first constructs the calibration-derived feature pool on English C4. It then evaluates zero-shot multiple-choice performance on the validation splits of:

```text
WinoGrande (winogrande_xl)
PIQA
HellaSwag
```

For each candidate answer, the code scores the continuation by its **average token log-probability**. It reports accuracy for the base model, unrestricted SAE proxy, and pool-restricted proxy, together with the corresponding performance drops.

Outputs:

```text
final_ops/zero_shot_base_vs_proxy_results_final.csv
final_ops/zero_shot_base_vs_proxy_summary_final.csv
```

`zero_shot_base_vs_proxy_results_final.csv` contains per-example predictions and correctness indicators.  
`zero_shot_base_vs_proxy_summary_final.csv` contains the model/layer/dataset-level accuracies used to form Table 7, including:

```text
Acc_M
Acc_SoM
Acc_hG
Drop_SoM
Drop_hG
P
```

The released summary contains 1,000 examples per dataset for each evaluated model/layer configuration.

To change the number of evaluation examples, models, or Llama layers, edit the `CONFIG` dictionary near the top of `downstream_performance.py`.

---

## 10. Table 8: frequency-thresholded pools

```bash
cd frequency

python llama_layer30_frequency_thresholded_pools.py
```

Default thresholds are:

```text
1, 2, 5, 10, 20, 50
```

The script counts how often each SAE feature is selected during calibration and defines

\[
G_\tau = \{j : n_j \ge \tau\}.
\]

It then evaluates the same held-out stream for each threshold.

Default outputs:

```text
frequency_threshold_layer30/feature_counts_layer_30.pt
frequency_threshold_layer30/feature_counts_layer_30.csv
frequency_threshold_layer30/pool_frequency_preview_layer_30.csv
frequency_threshold_layer30/frequency_threshold_certificates_layer_30.csv
```

Useful modes:

```bash
# Only compute calibration frequencies / pool-size preview
python llama_layer30_frequency_thresholded_pools.py --preview-only

# Reuse already saved feature counts
python llama_layer30_frequency_thresholded_pools.py --reuse-counts

# Custom thresholds
python llama_layer30_frequency_thresholded_pools.py \
  --thresholds 1 2 5 10 20 50 \
  --reuse-counts
```

The manuscript-style \(N^\star\) grid defaults to:

```text
10k, 30k, 50k, ...
```

and can be changed with `--nstar-start` and `--nstar-step`.

---

## Precomputed outputs

The repository intentionally includes many of the CSVs and rendered plots from the reported experiments. This allows the plotting/aggregation stages to be inspected without rerunning every large-model forward pass.

In particular:

```text
fig1_tab1/final_ops_*/
fig1_tab1/calibration_sensitivity/
fig2_tab2_tab3/bounds/
fig2_tab2_tab3/final_ops/
fig3/final_ops/
fig4/final_ops_final/
fig5/gpt2_layerwise_exact_final/
fig7/final_ops_corruption_updated/
tab6/final_ops/
frequency/frequency_threshold_layer30/
```

contain released outputs used in the analyses.

---

## Important reproducibility notes

### Fixed versus selected sample size

For any fixed, pre-specified held-out sample size \(N\), the certificate has the stated pointwise confidence guarantee.

The reported first crossing \(N^\star\) is used only as a **descriptive statistic**: it is the first tested sample size at which \(U_{\mathrm{SAE+Occam}} < R_{\mathrm{unif}}\). Because \(N^\star\) is selected after examining multiple nested sample sizes, it is not assigned a separate simultaneous \(1-\delta\) guarantee.

### Calibration and evaluation

The certification scripts construct the feature pool from calibration data that are disjoint from the held-out evaluation stream. Conditional on calibration, the selected pool and its pool-restricted predictor are fixed before evaluation.

### Model labels

Some source files retain the internal label `Llama-3.1-8B`, while the checkpoint actually loaded by the released scripts is:

```text
meta-llama/Meta-Llama-3-8B
```

The paper refers to this model as **Llama-3-8B**.

### Output filenames

Several filenames were kept from intermediate experimental versions. The directory and script configuration should be treated as the authoritative record of the actual setting. For example, the cross-model output filename contains `k=64` even when the script is run under another `topk`; the containing `final_ops_<k>/` directory records the actual Top-\(k\) run.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{sparse_lens_certification_2026,
  title   = {When Can We Trust the Sparse Lens? A Certification Framework for SAE Faithfulness},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2026}
}
```



---

## Repository

https://github.com/newcodevelop/sparse-lens-certification
