#!/usr/bin/env python3
"""Calibration-size sensitivity for the SAE certification experiments.

This script mirrors the model/SAE, Top-k, smoothed-loss, exact-complexity,
and fully empirical certificate logic of
``plot2_batched_topk_unified_gemma_gpt_llama_delta_exact_hoeffding_final.py``
while varying only the calibration budget.

Crucially, all calibration pools are nested prefixes of the same shuffled C4
calibration stream, and every pool is evaluated on the SAME held-out stream.
The held-out stream begins after the largest calibration budget, so changing
calibration size does not also change the evaluation data.

Default calibration budgets: 280k, 560k, 1.12M, 2.24M tokens.
Default evaluation budget: 320k tokens (N ~= 10,000 sequences of length 32).

Outputs one CSV table containing, for every model and calibration budget:
  - dictionary size m, pool size P, and P/m
  - exact ln C(m,P)
  - R_hat(M), R_hat(h_G), reconstruction gap, and mismatch rate
  - mismatch, Occam, and reconstruction concentration contributions
  - full structural certificate and its margin to the uniform baseline
  - direct Hoeffding bounds for M and h_G
  - changes in P, P/m, mismatch, and certificate from the previous budget
"""

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae


@dataclass
class ModelConfig:
    name: str
    sae_backend: str
    sae_release: str
    hook_name: str
    batch_size: int = 16
    sae_id: Optional[str] = None
    sae_hookpoint: Optional[str] = None
    dtype: torch.dtype = torch.float32


MODEL_CONFIGS = {
    "GPT-2 Small": ModelConfig(
        name="gpt2-small",
        sae_backend="sae_lens",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        hook_name="blocks.6.hook_resid_pre",
        batch_size=16,
        dtype=torch.float32,
    ),
    "Gemma-2B": ModelConfig(
        name="gemma-2b",
        sae_backend="sae_lens",
        sae_release="gemma-2b-res-jb",
        sae_id="blocks.12.hook_resid_post",
        hook_name="blocks.12.hook_resid_post",
        batch_size=16,
        dtype=torch.float32,
    ),
    "Llama-3.1-8B": ModelConfig(
        name="meta-llama/Meta-Llama-3-8B",
        sae_backend="sparsify",
        sae_release="EleutherAI/sae-llama-3-8b-32x",
        sae_hookpoint="layers.30",
        hook_name="blocks.30.hook_resid_post",
        batch_size=16,
        dtype=torch.bfloat16,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_budgets",
        type=int,
        nargs="+",
        default=[280_000, 560_000, 1_120_000, 2_240_000],
        help="Nested calibration-token budgets to compare.",
    )
    parser.add_argument(
        "--evaluation_tokens",
        type=int,
        default=320_000,
        help="Fixed held-out evaluation budget shared by all calibration pools.",
    )
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Lowercase confidence parameter delta; distinct from loss-range width capital Delta.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=32_000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Subset of model configurations to run.",
    )
    parser.add_argument("--out_dir", type=str, default="./calibration_sensitivity")
    return parser.parse_args()


ARGS = parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def exact_log_binom(m: int, p: int) -> float:
    """Return natural log ln(C(m,p)) stably using log-gamma."""
    m = int(m)
    p = int(p)
    if p < 0 or p > m:
        raise ValueError(f"p must satisfy 0 <= p <= m, got p={p}, m={m}")
    if p == 0 or p == m:
        return 0.0
    p = min(p, m - p)  # C(m,p) = C(m,m-p)
    return (
        math.lgamma(m + 1.0)
        - math.lgamma(p + 1.0)
        - math.lgamma(m - p + 1.0)
    )


def direct_hoeffding_bound(empirical_risk: float, range_width: float, delta: float, N: float):
    margin = range_width * math.sqrt(math.log(1.0 / delta) / (2.0 * N))
    return empirical_risk + margin, margin


def get_topk_sparse(activations: torch.Tensor, k: int):
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations: torch.Tensor, k: int):
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(-1, topk_inds, True)
    return activations * mask


def smoothed_bpd_loss(logits, tokens, alpha, vocab_size, reduction="mean"):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(
        probs_shifted, -1, tokens_shifted.unsqueeze(-1)
    ).squeeze(-1)
    smoothed_probs = (1.0 - alpha) * true_probs + (alpha / vocab_size)
    log_probs = -torch.log2(smoothed_probs)
    loss_per_seq = log_probs.mean(dim=-1)
    if reduction == "none":
        return loss_per_seq
    return loss_per_seq.mean()


def extract_pre_acts(encode_out):
    if torch.is_tensor(encode_out):
        return encode_out
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    if isinstance(encode_out, (list, tuple)) and len(encode_out) > 0:
        return encode_out[0]
    raise TypeError(f"Unsupported SAE encode output type: {type(encode_out)}")


class SAEAdapter:
    def __init__(self, sae, backend, device):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act):
        return extract_pre_acts(self.sae.encode(flat_act))

    def decode_topk(self, feature_acts_raw, k):
        if self.backend == "sparsify":
            top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)
            recons = self.sae.decode(top_acts, top_indices)
            active_mask = torch.zeros_like(feature_acts_raw, dtype=torch.bool)
            active_mask.scatter_(-1, top_indices, top_acts > 0)
            return recons, active_mask

        sparse_acts = apply_topk_dense(feature_acts_raw, k)
        recons = self.sae.decode(sparse_acts)
        active_mask = sparse_acts > 0
        return recons, active_mask

    def get_latent_dim(self, d_model, dtype):
        if self.backend == "sae_lens":
            return int(self.sae.cfg.d_sae)
        dummy = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return int(self.encode_pre_acts(dummy).shape[-1])


def load_model_and_sae(cfg: ModelConfig):
    model = HookedTransformer.from_pretrained(
        cfg.name, device=DEVICE, dtype=cfg.dtype
    )
    if cfg.sae_backend == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=cfg.sae_release,
            sae_id=cfg.sae_id,
            device=DEVICE,
        )
    elif cfg.sae_backend == "sparsify":
        sae = Sae.load_from_hub(
            cfg.sae_release,
            hookpoint=cfg.sae_hookpoint,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown SAE backend: {cfg.sae_backend}")
    sae.eval()
    return model, SAEAdapter(sae, cfg.sae_backend, DEVICE)


def token_stream(model, batch_size):
    """Infinite-ish deterministic shuffled C4 stream, yielding fixed-length batches."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=ARGS.shuffle_seed, buffer_size=ARGS.buffer_size)
    iterator = iter(dataset)

    while True:
        texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                texts.append(item["text"] if "text" in item else item["content"])
        except StopIteration:
            if not texts:
                break

        if not texts:
            break

        tokens = model.to_tokens(texts)
        if tokens.shape[1] < ARGS.seq_len:
            continue
        yield tokens[:, : ARGS.seq_len].to(DEVICE)


@torch.no_grad()
def build_nested_calibration_pools(model, adapter, cfg, budgets, m):
    """Build nested union-of-support pools from one calibration stream."""
    budgets = sorted(set(int(x) for x in budgets))
    if not budgets or budgets[0] <= 0:
        raise ValueError("Calibration budgets must be positive integers.")

    stream = token_stream(model, cfg.batch_size)
    active_indices = torch.zeros(m, dtype=torch.bool, device=DEVICE)
    pools = {}
    actual_tokens = {}
    next_idx = 0
    total_tokens = 0
    max_budget = budgets[-1]

    pbar = tqdm(total=max_budget, desc=f"Calibration ({cfg.name})")
    for tokens in stream:
        # Calibration semantics match the main code: process a batch whenever
        # the token count before that batch is below the maximum budget.
        if total_tokens >= max_budget:
            break

        _, cache = model.run_with_cache(tokens, names_filter=[cfg.hook_name])
        act = cache[cfg.hook_name]
        flat_act = act.reshape(-1, act.shape[-1])
        feature_acts_raw = adapter.encode_pre_acts(flat_act)
        _, active_mask = adapter.decode_topk(feature_acts_raw, ARGS.top_k)
        active_indices |= active_mask.any(dim=0)

        total_tokens += tokens.numel()
        pbar.update(tokens.numel())

        while next_idx < len(budgets) and total_tokens >= budgets[next_idx]:
            budget = budgets[next_idx]
            pools[budget] = active_indices.clone()
            actual_tokens[budget] = int(total_tokens)
            print(
                f"  calibration target={budget:,}, actual={total_tokens:,}: "
                f"P={int(active_indices.sum().item()):,}"
            )
            next_idx += 1

        if next_idx >= len(budgets):
            break

    pbar.close()
    if len(pools) != len(budgets):
        missing = [b for b in budgets if b not in pools]
        raise RuntimeError(f"Could not construct pools for budgets: {missing}")
    return pools, actual_tokens


def fixed_evaluation_stream(model, batch_size, skip_calibration_tokens):
    """Yield one fixed held-out stream beginning after the largest calibration prefix."""
    stream = token_stream(model, batch_size)
    skipped_tokens = 0
    for tokens in stream:
        # Mirror calibration partition exactly. Any batch whose starting token
        # count is below the max calibration budget belongs to calibration and
        # is skipped in evaluation, including the final crossing batch.
        if skipped_tokens < skip_calibration_tokens:
            skipped_tokens += tokens.numel()
            continue
        yield tokens


def run_model(model_key: str, cfg: ModelConfig):
    print(f"\n===== CALIBRATION SENSITIVITY: {model_key} =====")
    torch.cuda.empty_cache()
    gc.collect()

    model, adapter = load_model_and_sae(cfg)
    if cfg.hook_name not in model.hook_dict:
        raise ValueError(f"Hook name not found: {cfg.hook_name}")

    m = adapter.get_latent_dim(model.cfg.d_model, cfg.dtype)
    vocab_size = int(model.cfg.d_vocab)
    B = math.log2(vocab_size / ARGS.alpha)
    Delta = math.log2(1.0 + ((1.0 - ARGS.alpha) * vocab_size / ARGS.alpha))
    uniform_baseline = math.log2(vocab_size)

    budgets = sorted(set(int(x) for x in ARGS.calibration_budgets))
    max_budget = max(budgets)
    pools, actual_cal_tokens = build_nested_calibration_pools(
        model, adapter, cfg, budgets, m
    )

    # Shared evaluation statistics: M and unrestricted S o M do not depend on
    # the calibration pool, so compute them once on the common held-out stream.
    shared_loss_M_sum = 0.0
    shared_eps_sum = 0.0
    shared_count = 0

    per_budget = {
        b: {
            "loss_hG_sum": 0.0,
            "violation_sum": 0.0,
            "count": 0,
        }
        for b in budgets
    }

    eval_tokens = 0
    eval_gen = fixed_evaluation_stream(model, cfg.batch_size, max_budget)
    pbar = tqdm(total=ARGS.evaluation_tokens, desc=f"Evaluation ({model_key})")

    with torch.no_grad():
        for tokens in eval_gen:
            if eval_tokens >= ARGS.evaluation_tokens:
                break

            orig_logits, cache = model.run_with_cache(tokens, names_filter=[cfg.hook_name])
            loss_M = smoothed_bpd_loss(
                orig_logits, tokens, ARGS.alpha, vocab_size, reduction="none"
            )

            act = cache[cfg.hook_name]
            flat_act = act.reshape(-1, act.shape[-1])
            feature_acts_raw = adapter.encode_pre_acts(flat_act)

            recons_unrestricted, active_mask_unrestricted = adapter.decode_topk(
                feature_acts_raw, ARGS.top_k
            )
            recons_unrestricted = recons_unrestricted.reshape(act.shape)

            def hook_unrestricted(activations, hook):
                return recons_unrestricted

            logits_Som = model.run_with_hooks(
                tokens, fwd_hooks=[(cfg.hook_name, hook_unrestricted)]
            )
            loss_Som = smoothed_bpd_loss(
                logits_Som, tokens, ARGS.alpha, vocab_size, reduction="none"
            )
            gap = torch.abs(loss_M - loss_Som)

            shared_loss_M_sum += float(loss_M.sum().item())
            shared_eps_sum += float(gap.sum().item())
            shared_count += int(loss_M.numel())

            for budget in budgets:
                pool_mask = pools[budget]
                masked_acts = feature_acts_raw * pool_mask.unsqueeze(0)
                recons_restricted, _ = adapter.decode_topk(masked_acts, ARGS.top_k)
                recons_restricted = recons_restricted.reshape(act.shape)

                def hook_restricted(activations, hook, recons=recons_restricted):
                    return recons

                logits_hG = model.run_with_hooks(
                    tokens, fwd_hooks=[(cfg.hook_name, hook_restricted)]
                )
                loss_hG = smoothed_bpd_loss(
                    logits_hG, tokens, ARGS.alpha, vocab_size, reduction="none"
                )

                violation_mask = active_mask_unrestricted & (~pool_mask.unsqueeze(0))
                seq_has_violation = (
                    violation_mask.any(dim=-1)
                    .reshape(tokens.shape[0], -1)
                    .any(dim=-1)
                    .float()
                )

                per_budget[budget]["loss_hG_sum"] += float(loss_hG.sum().item())
                per_budget[budget]["violation_sum"] += float(seq_has_violation.sum().item())
                per_budget[budget]["count"] += int(loss_hG.numel())

            batch_tokens = tokens.numel()
            eval_tokens += batch_tokens
            pbar.update(batch_tokens)

            if eval_tokens >= ARGS.evaluation_tokens:
                break

    pbar.close()

    if shared_count == 0:
        raise RuntimeError("No evaluation samples were processed.")

    N = float(shared_count)
    R_hat_M = shared_loss_M_sum / shared_count
    eps_loss_hat = shared_eps_sum / shared_count
    direct_U_M, hoeffding_margin = direct_hoeffding_bound(
        R_hat_M, Delta, ARGS.delta, N
    )

    rows = []
    for budget in budgets:
        stats = per_budget[budget]
        if stats["count"] != shared_count:
            raise RuntimeError(
                f"Count mismatch for calibration budget {budget}: "
                f"{stats['count']} vs shared {shared_count}"
            )

        P = int(pools[budget].sum().item())
        P_over_m = P / m
        log_C = exact_log_binom(m, P)
        R_hat_hG = stats["loss_hG_sum"] / shared_count
        eta_hat = stats["violation_sum"] / shared_count

        mismatch_empirical = eta_hat * Delta
        eta_conc = Delta * math.sqrt(math.log(3.0 / ARGS.delta) / (2.0 * N))
        occam_term = Delta * math.sqrt(
            (log_C + math.log(3.0 / ARGS.delta)) / (2.0 * N)
        )
        eps_conc = Delta * math.sqrt(math.log(6.0 / ARGS.delta) / (2.0 * N))
        total_bound = (
            R_hat_hG
            + eps_loss_hat
            + mismatch_empirical
            + eta_conc
            + occam_term
            + eps_conc
        )
        direct_U_hG, _ = direct_hoeffding_bound(
            R_hat_hG, Delta, ARGS.delta, N
        )

        rows.append(
            {
                "Model": model_key,
                "Calibration_Tokens_Target": budget,
                "Calibration_Tokens_Actual": actual_cal_tokens[budget],
                "Evaluation_Tokens_Actual": int(eval_tokens),
                "N": int(N),
                "TopK": ARGS.top_k,
                "m": m,
                "P": P,
                "P_over_m": P_over_m,
                "exact_log_C_m_P": log_C,
                "R_hat_M": R_hat_M,
                "R_hat_hG": R_hat_hG,
                "eps_loss_hat": eps_loss_hat,
                "eta_hat": eta_hat,
                "Mismatch_Empirical_DeltaEta": mismatch_empirical,
                "Mismatch_Concentration": eta_conc,
                "Occam_Term": occam_term,
                "Reconstruction_Concentration": eps_conc,
                "Total_Bound": total_bound,
                "Uniform_Baseline": uniform_baseline,
                "Bound_minus_Uniform": total_bound - uniform_baseline,
                "Nonvacuous": bool(total_bound < uniform_baseline),
                "Direct_U_M": direct_U_M,
                "Direct_U_hG": direct_U_hG,
                "Hoeffding_Margin": hoeffding_margin,
                "B": B,
                "Delta": Delta,
                "delta": ARGS.delta,
            }
        )

    # Release large model before moving to the next configuration.
    del model, adapter, pools
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def add_sensitivity_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Model", "Calibration_Tokens_Target"]).reset_index(drop=True)
    for source, target in [
        ("P", "Delta_P_from_prev"),
        ("P_over_m", "Delta_P_over_m_from_prev"),
        ("eta_hat", "Delta_eta_from_prev"),
        ("R_hat_hG", "Delta_R_hat_hG_from_prev"),
        ("Total_Bound", "Delta_Total_Bound_from_prev"),
        ("exact_log_C_m_P", "Delta_log_C_from_prev"),
    ]:
        df[target] = df.groupby("Model")[source].diff()
    return df


all_rows = []
for model_name in ARGS.models:
    all_rows.extend(run_model(model_name, MODEL_CONFIGS[model_name]))

os.makedirs(ARGS.out_dir, exist_ok=True)
df = add_sensitivity_deltas(pd.DataFrame(all_rows))

out_csv = os.path.join(ARGS.out_dir, "calibration_size_sensitivity_table.csv")
df.to_csv(out_csv, index=False)

compact_cols = [
    "Model",
    "Calibration_Tokens_Target",
    "Calibration_Tokens_Actual",
    "N",
    "m",
    "P",
    "P_over_m",
    "eta_hat",
    "R_hat_hG",
    "eps_loss_hat",
    "exact_log_C_m_P",
    "Total_Bound",
    "Uniform_Baseline",
    "Bound_minus_Uniform",
    "Direct_U_M",
    "Direct_U_hG",
    "Delta_P_from_prev",
    "Delta_eta_from_prev",
    "Delta_Total_Bound_from_prev",
]

print("\n===== CALIBRATION-SIZE SENSITIVITY TABLE =====")
print(df[compact_cols].to_string(index=False))
print(f"\nSaved full sensitivity table to: {out_csv}")
