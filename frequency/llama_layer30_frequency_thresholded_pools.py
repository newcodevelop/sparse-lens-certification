#!/usr/bin/env python3
"""
Frequency-thresholded concept-pool experiment for the Layer-30 Llama SAE.

Purpose
-------
Implements the Reviewer-C diagnostic:

    fixed-calibration experiment using frequency-thresholded pools
    to distinguish frequently reused features from a long tail of
    rare features.

Adapted from:
    plot2_batched_llama_layered_delta_exact_hoeffding_final(2).py

The calibration stream is fixed (2.24M tokens by default). For each SAE
feature j, we count how many calibration token positions select that
feature under the same Top-k-plus-ReLU rule used in the source script:

    n_j = number of calibration token positions where feature j is active.

For a threshold tau we define

    G_tau = {j : n_j >= tau}.

tau=1 exactly recovers the original union-support pool.

For each threshold, on the SAME held-out stream, the script reports:
    - P and P/m
    - exact ln C(m,P)
    - empirical support-mismatch rate eta_hat
    - empirical restricted-proxy risk R_hat(h_G)
    - empirical reconstruction-loss gap eps_loss_hat
    - direct Hoeffding bounds for M and h_G
    - fixed-pool certificate U_SAE
    - exact-P extension U_SAE+Occam
    - common asymptotic floor
    - manuscript-style N* (first 10k,30k,50k,... grid point at which
      U_SAE+Occam becomes non-vacuous, when the asymptotic floor permits)
    - R_hat(h_G), U_SAE, and U_SAE+Occam evaluated at that N*
    - uninformed baseline log2(V)

Important implementation detail
-------------------------------
This intentionally preserves the coding rule of the source script:
raw SAE encoder pre-activations are ranked with torch.topk and ReLU is
applied only to the selected values. Thus "frequency" counts positive
selected features after this Top-k-plus-ReLU procedure.

The source script loads:
    meta-llama/Meta-Llama-3-8B
    EleutherAI/sae-llama-3-8b-32x
"""

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sparsify import Sae
from tqdm import tqdm
from transformer_lens import HookedTransformer


# ============================================================
# 1. ARGUMENTS / CONFIGURATION
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-calibration frequency-thresholded pool experiment."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=30,
        help="Transformer layer index. Default: 30.",
    )
    parser.add_argument(
        "--calibration-tokens",
        type=int,
        default=2_240_000,
        help="Number of calibration tokens. Default: 2.24M.",
    )
    parser.add_argument(
        "--evaluation-tokens",
        type=int,
        default=320_000,
        help=(
            "Number of held-out evaluation tokens. With sequence length 32, "
            "320,000 tokens = N=10,000 evaluation sequences."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size. Default: 16.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="Frequency thresholds tau to evaluate.",
    )
    parser.add_argument(
        "--preview-thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        help=(
            "Cheap calibration-only threshold preview. "
            "These do not trigger evaluation unless also in --thresholds."
        ),
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Stop after calibration frequency counting and pool-size preview.",
    )
    parser.add_argument(
        "--reuse-counts",
        action="store_true",
        help="Reuse saved calibration feature counts if available.",
    )
    parser.add_argument(
        "--nstar-start",
        type=int,
        default=10_000,
        help=(
            "First N on the manuscript-style N* grid. "
            "Default: 10,000 sequences."
        ),
    )
    parser.add_argument(
        "--nstar-step",
        type=int,
        default=20_000,
        help=(
            "Spacing of the manuscript-style N* grid. "
            "Default: 20,000 sequences (10k, 30k, 50k, ...)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./frequency_threshold_layer30",
        help="Directory for CSV/PT outputs.",
    )
    return parser.parse_args()


ARGS = parse_args()
SELECTED_LAYER = ARGS.layer

MODEL_LABEL = "Llama-3-8B"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x"
SAE_HOOKPOINT = f"layers.{SELECTED_LAYER}"
HOOK_NAME = f"blocks.{SELECTED_LAYER}.hook_resid_post"

SEQUENCE_LENGTH = 32
TOP_K = 64
ALPHA = 0.5
CONFIDENCE_DELTA = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTS_PATH = OUTPUT_DIR / f"feature_counts_layer_{SELECTED_LAYER}.pt"
FEATURE_CSV_PATH = OUTPUT_DIR / f"feature_counts_layer_{SELECTED_LAYER}.csv"
PREVIEW_CSV_PATH = OUTPUT_DIR / f"pool_frequency_preview_layer_{SELECTED_LAYER}.csv"
RESULTS_CSV_PATH = OUTPUT_DIR / f"frequency_threshold_certificates_layer_{SELECTED_LAYER}.csv"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def exact_log_binom(m: int, p: int) -> float:
    """Return ln(C(m,p)) stably using log-gamma."""
    m = int(m)
    p = int(p)
    if p < 0 or p > m:
        raise ValueError(f"p must satisfy 0 <= p <= m, got p={p}, m={m}")
    if p == 0 or p == m:
        return 0.0
    p_small = min(p, m - p)
    return (
        math.lgamma(m + 1.0)
        - math.lgamma(p_small + 1.0)
        - math.lgamma(m - p_small + 1.0)
    )


def direct_hoeffding_bound(
    empirical_risk: float,
    range_width: float,
    delta: float,
    N: float,
):
    """Direct one-sided fixed-predictor Hoeffding upper bound."""
    margin = range_width * math.sqrt(math.log(1.0 / delta) / (2.0 * N))
    return empirical_risk + margin, margin


def get_topk_sparse(activations: torch.Tensor, k: int):
    """
    Preserve the source script's coding rule:
      1) rank raw SAE pre-activations by VALUE using torch.topk;
      2) apply ReLU after selection.
    """
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def extract_pre_acts(encode_out):
    """Safely extract dense SAE pre-activations from sparsify output."""
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    return encode_out[0]


def smoothed_bpd_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    alpha: float,
    vocab_size: int,
    reduction: str = "mean",
):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]

    true_probs = torch.gather(
        probs_shifted,
        -1,
        tokens_shifted.unsqueeze(-1),
    ).squeeze(-1)

    smoothed_probs = (1.0 - alpha) * true_probs + (alpha / vocab_size)
    token_losses = -torch.log2(smoothed_probs)
    loss_per_seq = token_losses.mean(dim=-1)

    if reduction == "mean":
        return loss_per_seq.mean()
    return loss_per_seq


def get_tokens_generator(
    model,
    batch_size: int,
    device: str,
    mode: str,
    calibration_limit: int,
):
    """
    Reproduce the source script's C4 partitioning:
      - same shuffled C4 stream, seed 42;
      - calibration consumes the first calibration_limit tokens;
      - evaluation skips that prefix and begins afterward.
    """
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=32000)

    iterator = iter(dataset)
    tokens_processed_global = 0
    skip_needed = mode == "evaluation"

    while True:
        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item["text"] if "text" in item else item["content"]
                batch_texts.append(text)
        except StopIteration:
            if not batch_texts:
                break

        if not batch_texts:
            break

        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] < SEQUENCE_LENGTH:
            continue

        tokens = tokens[:, :SEQUENCE_LENGTH]
        num_tok = tokens.numel()

        if skip_needed:
            if tokens_processed_global < calibration_limit:
                tokens_processed_global += num_tok
                continue
            skip_needed = False

        elif mode == "calibration":
            if tokens_processed_global >= calibration_limit:
                break
            tokens_processed_global += num_tok

        yield tokens.to(device)


# ============================================================
# 3. CALIBRATION: FEATURE-FREQUENCY COUNTS
# ============================================================

def measure_feature_frequencies(
    model,
    sae,
    token_gen,
    hook_name: str,
    target_tokens: int,
    device: str,
):
    """
    For each feature j, count the number of calibration token positions
    where j survives Top-k and has a strictly positive post-ReLU value.

    Therefore G_{tau=1} exactly recovers the original union-support pool.
    """
    print(
        f"\n[Calibration] Counting feature frequencies on "
        f"{target_tokens:,} tokens with Top-k={TOP_K}..."
    )

    dummy_act = torch.zeros(
        1,
        model.cfg.d_model,
        device=device,
        dtype=DTYPE,
    )
    d_sae = extract_pre_acts(sae.encode(dummy_act)).shape[-1]

    feature_counts = torch.zeros(
        d_sae,
        dtype=torch.long,
        device=device,
    )

    total_tokens = 0
    pbar = tqdm(total=target_tokens, desc="Calibration frequency counts")

    for tokens in token_gen:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])

            feature_acts_raw = extract_pre_acts(sae.encode(flat_act))
            top_acts, top_indices = get_topk_sparse(feature_acts_raw, TOP_K)

            flat_idx = top_indices.reshape(-1)
            flat_is_active = (top_acts > 0).reshape(-1)
            active_idx = flat_idx[flat_is_active]

            if active_idx.numel() > 0:
                feature_counts.scatter_add_(
                    0,
                    active_idx,
                    torch.ones_like(active_idx, dtype=torch.long),
                )

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)

        if total_tokens >= target_tokens:
            break

    pbar.close()

    feature_counts_cpu = feature_counts.detach().cpu()
    union_P = int((feature_counts_cpu >= 1).sum().item())

    print(f"[Calibration] SAE dictionary size m = {d_sae:,}")
    print(f"[Calibration] tau=1 union pool P = {union_P:,}")
    print(f"[Calibration] tau=1 pool coverage P/m = {union_P / d_sae:.6f}")

    return feature_counts_cpu, d_sae


def save_feature_counts(feature_counts: torch.Tensor):
    torch.save(feature_counts, COUNTS_PATH)

    feature_df = pd.DataFrame(
        {
            "feature_index": np.arange(feature_counts.numel(), dtype=np.int64),
            "calibration_active_count": feature_counts.numpy(),
        }
    )
    feature_df.to_csv(FEATURE_CSV_PATH, index=False)

    print(f"Saved counts tensor: {COUNTS_PATH}")
    print(f"Saved per-feature counts CSV: {FEATURE_CSV_PATH}")


def make_pool_preview(
    feature_counts: torch.Tensor,
    m: int,
    thresholds,
):
    rows = []

    total_active_occurrences = int(feature_counts.sum().item())
    nonzero_counts = feature_counts[feature_counts > 0]

    print("\nCalibration frequency summary")
    print("--------------------------------")
    print(f"Total positive Top-k feature occurrences: {total_active_occurrences:,}")
    print(f"Features seen at least once: {(feature_counts > 0).sum().item():,} / {m:,}")

    if nonzero_counts.numel() > 0:
        quantiles = torch.quantile(
            nonzero_counts.float(),
            torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]),
        )
        print(
            "Positive-count quantiles "
            "(min,25%,50%,75%,90%,99%,max): "
            + ", ".join(f"{x.item():.1f}" for x in quantiles)
        )

    print("\nPool-size preview")
    print("-----------------")
    for tau in sorted(set(int(t) for t in thresholds)):
        if tau < 1:
            raise ValueError(f"All thresholds must be >= 1, got {tau}")

        pool_mask = feature_counts >= tau
        P = int(pool_mask.sum().item())
        P_over_m = P / m
        log_c = exact_log_binom(m, P)

        rows.append(
            {
                "tau": tau,
                "P": P,
                "m": m,
                "P_over_m": P_over_m,
                "exact_log_C_m_P": log_c,
            }
        )

        print(
            f"tau={tau:>5d} | P={P:>7,d} | "
            f"P/m={P_over_m:.6f} | lnC(m,P)={log_c:.3f}"
        )

    preview_df = pd.DataFrame(rows)
    preview_df.to_csv(PREVIEW_CSV_PATH, index=False)
    print(f"\nSaved preview CSV: {PREVIEW_CSV_PATH}")
    return preview_df


# ============================================================
# 4. CERTIFICATE FORMULAS
# ============================================================

def compute_certificates(
    R_hat_hG: float,
    eps_loss_hat: float,
    eta_hat: float,
    Delta: float,
    delta: float,
    N: int,
    m: int,
    P: int,
):
    """
    Fixed-pool certificate:
        U_SAE =
            Rhat(h_G) + epshat + Delta*etahat
            + 2*Delta*sqrt(ln(3/delta)/(2N))
            + Delta*sqrt(ln(6/delta)/(2N))

    Exact-P extension:
        U_SAE+Occam =
            Rhat(h_G) + epshat + Delta*etahat
            + Delta*sqrt(ln(3/delta)/(2N))
            + Delta*sqrt((ln C(m,P)+ln(3/delta))/(2N))
            + Delta*sqrt(ln(6/delta)/(2N))
    """
    mismatch_conc = Delta * math.sqrt(math.log(3.0 / delta) / (2.0 * N))
    proxy_fixed_conc = mismatch_conc
    eps_conc = Delta * math.sqrt(math.log(6.0 / delta) / (2.0 * N))

    log_c = exact_log_binom(m, P)
    proxy_occam_conc = Delta * math.sqrt(
        (log_c + math.log(3.0 / delta)) / (2.0 * N)
    )

    empirical_core = R_hat_hG + eps_loss_hat + Delta * eta_hat

    U_SAE = (
        empirical_core
        + proxy_fixed_conc
        + mismatch_conc
        + eps_conc
    )

    U_SAE_Occam = (
        empirical_core
        + mismatch_conc
        + proxy_occam_conc
        + eps_conc
    )

    return {
        "exact_log_C_m_P": log_c,
        "empirical_core": empirical_core,
        "mismatch_conc": mismatch_conc,
        "proxy_fixed_conc": proxy_fixed_conc,
        "proxy_occam_conc": proxy_occam_conc,
        "eps_conc": eps_conc,
        "U_SAE": U_SAE,
        "U_SAE_Occam": U_SAE_Occam,
    }



def project_nstar_on_grid(
    R_hat_hG: float,
    eps_loss_hat: float,
    eta_hat: float,
    Delta: float,
    delta: float,
    m: int,
    P: int,
    R_unif: float,
    grid_start: int = 10_000,
    grid_step: int = 20_000,
):
    """
    Manuscript-style projected N* for U_SAE+Occam.

    The empirical quantities R_hat(h_G), eps_loss_hat, and eta_hat are
    held fixed at the values measured on the current held-out stream.
    Only the explicit N-dependent concentration terms are varied.

    This matches the continuation logic used in the manuscript's
    layerwise analysis: if the asymptotic floor is below R_unif, find
    the first point on the grid

        grid_start, grid_start + grid_step, ...

    at which U_SAE+Occam < R_unif.

    If the asymptotic floor is >= R_unif, no finite N can make the
    certificate strictly non-vacuous, so N* is reported as None.

    Returns both certificates at N* plus the common asymptotic floor.
    """
    if grid_start <= 0:
        raise ValueError("grid_start must be positive.")
    if grid_step <= 0:
        raise ValueError("grid_step must be positive.")

    asymptotic_floor = R_hat_hG + eps_loss_hat + Delta * eta_hat

    # Strict non-vacuity requires U < R_unif.  If the floor is equal
    # to or above R_unif, positive concentration terms prevent crossing.
    if asymptotic_floor >= R_unif:
        return {
            "Asymptotic_floor": asymptotic_floor,
            "N_star": None,
            "R_hat_hG_at_Nstar": np.nan,
            "U_SAE_at_Nstar": np.nan,
            "U_SAE_Occam_at_Nstar": np.nan,
            "N_star_status": "Never",
        }

    log_c = exact_log_binom(m, P)

    # U_SAE+Occam(N) = floor + coeff_occam / sqrt(N).
    coeff_occam = Delta * (
        math.sqrt(math.log(3.0 / delta) / 2.0)
        + math.sqrt((log_c + math.log(3.0 / delta)) / 2.0)
        + math.sqrt(math.log(6.0 / delta) / 2.0)
    )

    gap_to_baseline = R_unif - asymptotic_floor

    # Continuous threshold at which equality would hold. Because the
    # certificate must be STRICTLY below R_unif, the selected grid point
    # must lie strictly above this threshold when equality is exact.
    continuous_required_N = (coeff_occam / gap_to_baseline) ** 2

    if grid_start > continuous_required_N:
        N_star = int(grid_start)
    else:
        # First manuscript-grid point strictly greater than the equality
        # threshold.
        steps = math.floor((continuous_required_N - grid_start) / grid_step) + 1
        N_star = int(grid_start + steps * grid_step)

    # Numerical guard: verify strict crossing and move one grid step if
    # floating-point roundoff happens to place us on/above the baseline.
    cert_at_nstar = compute_certificates(
        R_hat_hG=R_hat_hG,
        eps_loss_hat=eps_loss_hat,
        eta_hat=eta_hat,
        Delta=Delta,
        delta=delta,
        N=N_star,
        m=m,
        P=P,
    )
    while cert_at_nstar["U_SAE_Occam"] >= R_unif:
        N_star += int(grid_step)
        cert_at_nstar = compute_certificates(
            R_hat_hG=R_hat_hG,
            eps_loss_hat=eps_loss_hat,
            eta_hat=eta_hat,
            Delta=Delta,
            delta=delta,
            N=N_star,
            m=m,
            P=P,
        )

    return {
        "Asymptotic_floor": asymptotic_floor,
        "N_star": N_star,
        # Empirical terms are frozen for the projection, so R_hat(h_G)
        # at projected N* is the measured held-out value.
        "R_hat_hG_at_Nstar": R_hat_hG,
        "U_SAE_at_Nstar": cert_at_nstar["U_SAE"],
        "U_SAE_Occam_at_Nstar": cert_at_nstar["U_SAE_Occam"],
        "N_star_status": "Projected_from_fixed_empirical_terms",
    }


# ============================================================
# 5. HELD-OUT EVALUATION
# ============================================================

def evaluate_frequency_thresholds(
    model,
    sae,
    feature_counts: torch.Tensor,
    thresholds,
    eval_gen,
    hook_name: str,
    target_eval_tokens: int,
    vocab_size: int,
    m: int,
):
    """
    Evaluate every requested G_tau on the same held-out stream.

    Base-model loss and unrestricted SAE loss are computed once per batch.
    Each tau then requires one restricted-proxy forward pass.
    """
    thresholds = sorted(set(int(t) for t in thresholds))
    if any(t < 1 for t in thresholds):
        raise ValueError("All thresholds must satisfy tau >= 1.")

    pool_masks = {}
    pool_sizes = {}
    for tau in thresholds:
        mask_cpu = feature_counts >= tau
        pool_sizes[tau] = int(mask_cpu.sum().item())
        pool_masks[tau] = mask_cpu.to(DEVICE)

    loss_M_all = []
    eps_loss_all = []

    threshold_results = {
        tau: {
            "loss_hG": [],
            "pool_violation": [],
        }
        for tau in thresholds
    }

    total_tokens = 0
    pbar = tqdm(total=target_eval_tokens, desc="Held-out threshold evaluation")

    for tokens in eval_gen:
        with torch.no_grad():
            # A. Original model
            orig_logits, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )
            loss_M = smoothed_bpd_loss(
                orig_logits,
                tokens,
                ALPHA,
                vocab_size,
                reduction="none",
            )

            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])
            feature_acts_raw = extract_pre_acts(sae.encode(flat_act))

            # B. Unrestricted SAE proxy S o M
            top_acts_unrestricted, top_indices_unrestricted = get_topk_sparse(
                feature_acts_raw,
                TOP_K,
            )
            recons_unrestricted = sae.decode(
                top_acts_unrestricted,
                top_indices_unrestricted,
            ).reshape(act.shape)

            def hook_unrestricted(_activations, hook):
                return recons_unrestricted

            logits_Som = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_unrestricted)],
            )
            loss_Som = smoothed_bpd_loss(
                logits_Som,
                tokens,
                ALPHA,
                vocab_size,
                reduction="none",
            )

            gap = torch.abs(loss_M - loss_Som)

            loss_M_all.extend(loss_M.detach().cpu().float().numpy())
            eps_loss_all.extend(gap.detach().cpu().float().numpy())

            # C. Restricted h_{G_tau}
            for tau in thresholds:
                pool_mask = pool_masks[tau]

                masked_acts = feature_acts_raw * pool_mask.unsqueeze(0)
                top_acts_restricted, top_indices_restricted = get_topk_sparse(
                    masked_acts,
                    TOP_K,
                )

                recons_restricted = sae.decode(
                    top_acts_restricted,
                    top_indices_restricted,
                ).reshape(act.shape)

                # Sequence-level mismatch, preserving the source script.
                violation_mask_sparse = (
                    (top_acts_unrestricted > 0)
                    & (~pool_mask[top_indices_unrestricted])
                )
                seq_has_violation = (
                    violation_mask_sparse
                    .any(dim=-1)
                    .reshape(tokens.shape[0], -1)
                    .any(dim=-1)
                    .float()
                )

                def hook_restricted(_activations, hook, recon=recons_restricted):
                    return recon

                logits_hG = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, hook_restricted)],
                )
                loss_hG = smoothed_bpd_loss(
                    logits_hG,
                    tokens,
                    ALPHA,
                    vocab_size,
                    reduction="none",
                )

                threshold_results[tau]["loss_hG"].extend(
                    loss_hG.detach().cpu().float().numpy()
                )
                threshold_results[tau]["pool_violation"].extend(
                    seq_has_violation.detach().cpu().float().numpy()
                )

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)

        if total_tokens >= target_eval_tokens:
            break

    pbar.close()

    N = int(total_tokens // SEQUENCE_LENGTH)
    if N <= 0:
        raise RuntimeError("No evaluation sequences were processed.")

    R_hat_M = float(np.mean(loss_M_all))
    eps_loss_hat = float(np.mean(eps_loss_all))

    B = math.log2(vocab_size / ALPHA)
    Delta = math.log2(
        1.0 + ((1.0 - ALPHA) * vocab_size / ALPHA)
    )
    R_unif = math.log2(vocab_size)

    direct_U_M, hoeffding_margin = direct_hoeffding_bound(
        R_hat_M,
        Delta,
        CONFIDENCE_DELTA,
        N,
    )

    rows = []

    print("\nFinal fixed-calibration frequency-threshold results")
    print("====================================================")
    print(
        f"Evaluation: {total_tokens:,} tokens = N={N:,} sequences | "
        f"Rhat(M)={R_hat_M:.6f} | epshat={eps_loss_hat:.6f}"
    )

    for tau in thresholds:
        P = pool_sizes[tau]
        P_over_m = P / m

        R_hat_hG = float(np.mean(threshold_results[tau]["loss_hG"]))
        eta_hat = float(np.mean(threshold_results[tau]["pool_violation"]))

        cert = compute_certificates(
            R_hat_hG=R_hat_hG,
            eps_loss_hat=eps_loss_hat,
            eta_hat=eta_hat,
            Delta=Delta,
            delta=CONFIDENCE_DELTA,
            N=N,
            m=m,
            P=P,
        )

        direct_U_hG, _ = direct_hoeffding_bound(
            R_hat_hG,
            Delta,
            CONFIDENCE_DELTA,
            N,
        )

        nstar_info = project_nstar_on_grid(
            R_hat_hG=R_hat_hG,
            eps_loss_hat=eps_loss_hat,
            eta_hat=eta_hat,
            Delta=Delta,
            delta=CONFIDENCE_DELTA,
            m=m,
            P=P,
            R_unif=R_unif,
            grid_start=ARGS.nstar_start,
            grid_step=ARGS.nstar_step,
        )

        # If the measured current evaluation point itself is exactly the
        # first grid point and is already non-vacuous, this crossing is
        # directly observed rather than extrapolated.
        if (
            nstar_info["N_star"] is not None
            and nstar_info["N_star"] == N
            and cert["U_SAE_Occam"] < R_unif
        ):
            nstar_info["N_star_status"] = "Observed_at_current_evaluation_N"

        row = {
            "Model": MODEL_LABEL,
            "Layer": SELECTED_LAYER,
            "Hook": HOOK_NAME,
            "SAE_release": SAE_RELEASE,
            "Top_k": TOP_K,
            "Calibration_tokens": ARGS.calibration_tokens,
            "Evaluation_tokens": total_tokens,
            "N_sequences": N,
            "tau": tau,
            "m": m,
            "P": P,
            "P_over_m": P_over_m,
            "exact_log_C_m_P": cert["exact_log_C_m_P"],
            "R_hat_M": R_hat_M,
            "R_hat_hG": R_hat_hG,
            "eps_loss_hat": eps_loss_hat,
            "eta_hat": eta_hat,
            "Delta": Delta,
            "B": B,
            "delta": CONFIDENCE_DELTA,
            "Direct_U_M": direct_U_M,
            "Direct_U_hG": direct_U_hG,
            "Hoeffding_Margin": hoeffding_margin,
            "empirical_core": cert["empirical_core"],
            "mismatch_conc": cert["mismatch_conc"],
            "proxy_fixed_conc": cert["proxy_fixed_conc"],
            "proxy_occam_conc": cert["proxy_occam_conc"],
            "eps_conc": cert["eps_conc"],
            "U_SAE": cert["U_SAE"],
            "U_SAE_Occam": cert["U_SAE_Occam"],
            "R_unif": R_unif,
            "U_SAE_nonvacuous": cert["U_SAE"] < R_unif,
            "U_SAE_Occam_nonvacuous": cert["U_SAE_Occam"] < R_unif,
            "Asymptotic_floor": nstar_info["Asymptotic_floor"],
            "N_star": nstar_info["N_star"],
            "N_star_status": nstar_info["N_star_status"],
            "R_hat_hG_at_Nstar": nstar_info["R_hat_hG_at_Nstar"],
            "U_SAE_at_Nstar": nstar_info["U_SAE_at_Nstar"],
            "U_SAE_Occam_at_Nstar": nstar_info["U_SAE_Occam_at_Nstar"],
        }
        rows.append(row)

        if nstar_info["N_star"] is None:
            nstar_text = (
                f"Asymptotic={nstar_info['Asymptotic_floor']:.6f} | "
                "N*=Never"
            )
        else:
            nstar_text = (
                f"Asymptotic={nstar_info['Asymptotic_floor']:.6f} | "
                f"N*={int(nstar_info['N_star']):,} | "
                f"Rhat(hG)@N*={nstar_info['R_hat_hG_at_Nstar']:.6f} | "
                f"U_SAE@N*={nstar_info['U_SAE_at_Nstar']:.6f} | "
                f"U_SAE+Occam@N*={nstar_info['U_SAE_Occam_at_Nstar']:.6f}"
            )

        print(
            f"tau={tau:>5d} | "
            f"P={P:>7,d} ({P_over_m:.6f}) | "
            f"eta={eta_hat:.6f} | "
            f"Rhat(hG)={R_hat_hG:.6f} | "
            f"U_SAE={cert['U_SAE']:.6f} | "
            f"U_SAE+Occam={cert['U_SAE_Occam']:.6f} | "
            f"R_unif={R_unif:.6f} | "
            f"{nstar_text}"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved final results CSV: {RESULTS_CSV_PATH}")

    return results_df


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("============================================================")
    print("Frequency-thresholded pool experiment")
    print("============================================================")
    print(f"Model checkpoint : {MODEL_NAME}")
    print(f"SAE release      : {SAE_RELEASE}")
    print(f"Layer / hook     : {SELECTED_LAYER} / {HOOK_NAME}")
    print(f"Top-k            : {TOP_K}")
    print(f"Calibration      : {ARGS.calibration_tokens:,} tokens")
    print(
        f"Evaluation       : {ARGS.evaluation_tokens:,} tokens "
        f"(target N={ARGS.evaluation_tokens // SEQUENCE_LENGTH:,} sequences)"
    )
    print(f"Thresholds       : {ARGS.thresholds}")
    print(
        f"N* grid          : {ARGS.nstar_start:,}, "
        f"{ARGS.nstar_start + ARGS.nstar_step:,}, "
        f"{ARGS.nstar_start + 2 * ARGS.nstar_step:,}, ..."
    )
    print(f"Device           : {DEVICE}")
    print("============================================================")

    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nLoading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
    )

    print(
        f"Loading SAE from {SAE_RELEASE} "
        f"at hookpoint {SAE_HOOKPOINT}..."
    )
    sae = Sae.load_from_hub(
        SAE_RELEASE,
        hookpoint=SAE_HOOKPOINT,
    )
    sae = sae.to(DEVICE)
    sae.eval()

    if HOOK_NAME not in model.hook_dict:
        raise ValueError(f"Hook name not found in model.hook_dict: {HOOK_NAME}")

    dummy_act = torch.zeros(
        1,
        model.cfg.d_model,
        device=DEVICE,
        dtype=DTYPE,
    )
    m = int(extract_pre_acts(sae.encode(dummy_act)).shape[-1])
    print(f"SAE dictionary size m = {m:,}")

    # Calibration frequency counts
    if ARGS.reuse_counts and COUNTS_PATH.exists():
        print(f"\nReusing saved feature counts: {COUNTS_PATH}")
        feature_counts = torch.load(COUNTS_PATH, map_location="cpu").long()

        if feature_counts.numel() != m:
            raise ValueError(
                f"Saved counts have length {feature_counts.numel()}, "
                f"but current SAE has m={m}."
            )
    else:
        cal_gen = get_tokens_generator(
            model=model,
            batch_size=ARGS.batch_size,
            device=DEVICE,
            mode="calibration",
            calibration_limit=ARGS.calibration_tokens,
        )

        feature_counts, measured_m = measure_feature_frequencies(
            model=model,
            sae=sae,
            token_gen=cal_gen,
            hook_name=HOOK_NAME,
            target_tokens=ARGS.calibration_tokens,
            device=DEVICE,
        )

        if measured_m != m:
            raise RuntimeError(
                f"Dictionary-size mismatch: measured {measured_m}, expected {m}"
            )

        save_feature_counts(feature_counts)

    # Cheap pool-size preview.
    all_preview_thresholds = sorted(
        set(ARGS.preview_thresholds) | set(ARGS.thresholds)
    )
    make_pool_preview(
        feature_counts=feature_counts,
        m=m,
        thresholds=all_preview_thresholds,
    )

    if ARGS.preview_only:
        print("\n--preview-only supplied; stopping before held-out evaluation.")
        return

    # Same held-out stream for every threshold.
    eval_gen = get_tokens_generator(
        model=model,
        batch_size=ARGS.batch_size,
        device=DEVICE,
        mode="evaluation",
        calibration_limit=ARGS.calibration_tokens,
    )

    evaluate_frequency_thresholds(
        model=model,
        sae=sae,
        feature_counts=feature_counts,
        thresholds=ARGS.thresholds,
        eval_gen=eval_gen,
        hook_name=HOOK_NAME,
        target_eval_tokens=ARGS.evaluation_tokens,
        vocab_size=model.cfg.d_vocab,
        m=m,
    )


if __name__ == "__main__":
    main()
