import gc
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae


CONFIG = {
    "CALIBRATION_TOKENS": 2_240_000,
    "TEST_TOKENS": 2_240_000,
    "BATCH_SIZE_DEFAULT": 16,
    "CONTEXT_LEN": 32,
    "ALPHA": 0.5,
    "DELTA_CONF": 0.05,
    "TOP_K": 64,
    "SLIGHT_OOD_CORRUPTION_RATE": 0.15,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TOKENIZER_BUFFER_SIZE": 10_000,
    "OUTPUT_DIR": "./final_ops_corruption_updated",
    "CALIBRATION_SEED": 42,
    "EVALUATION_STREAM_SEED": 1042,
    "CORRUPTION_SEED": 43,
    "RANDOM_NOISE_SEED": 44,
    "CONDITIONS": [
        {"name": "English (IID)", "short_name": "IID", "mode": "english"},
        {"name": "Corrupted English (Slight-OOD)", "short_name": "Slight-OOD", "mode": "corrupted_english"},
        {"name": "Random Noise (Far-OOD)", "short_name": "Far-OOD", "mode": "random_noise"},
    ],
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16,
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "batch_size": 16,
        },
        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_backend": "sparsify",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "sae_hookpoint": "layers.30",
            "hook_name": "blocks.30.hook_resid_post",
            "batch_size": 16,
            "dtype": torch.bfloat16,
        },
    },
}


def smoothed_bpd_loss(logits, tokens, alpha, vocab_size, reduction="mean"):
    probs = torch.softmax(logits.float(), dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(probs_shifted, -1, tokens_shifted.unsqueeze(-1)).squeeze(-1)
    smoothed_probs = (1.0 - alpha) * true_probs + alpha / vocab_size
    loss_per_seq = (-torch.log2(smoothed_probs)).mean(dim=-1)
    return loss_per_seq.mean() if reduction == "mean" else loss_per_seq


def build_text_iterator(seed: int):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=CONFIG["TOKENIZER_BUFFER_SIZE"])
    return iter(ds)


def get_clean_text_batch(model, batch_size: int, iterator) -> Optional[torch.Tensor]:
    batch_tokens = []
    try:
        while len(batch_tokens) < batch_size:
            item = next(iterator)
            text = item.get("text", item.get("content", ""))
            tokens = model.to_tokens(text)[:, : CONFIG["CONTEXT_LEN"]]
            if tokens.shape[1] < CONFIG["CONTEXT_LEN"]:
                continue
            batch_tokens.append(tokens)
    except StopIteration:
        if not batch_tokens:
            return None
    if not batch_tokens:
        return None
    return torch.cat(batch_tokens, dim=0).to(CONFIG["DEVICE"])


def make_generator(seed: int):
    device_type = "cuda" if CONFIG["DEVICE"].startswith("cuda") else "cpu"
    g = torch.Generator(device=device_type)
    g.manual_seed(seed)
    return g


def apply_evaluation_condition(clean_tokens, mode, vocab_size, generator):
    if mode == "english":
        return clean_tokens
    if mode == "corrupted_english":
        corrupted = clean_tokens.clone()
        mask = torch.rand(corrupted.shape, device=corrupted.device, generator=generator) < CONFIG["SLIGHT_OOD_CORRUPTION_RATE"]
        if corrupted.shape[1] > 0:
            mask[:, 0] = False
        random_tokens = torch.randint(0, vocab_size, corrupted.shape, device=corrupted.device, dtype=corrupted.dtype, generator=generator)
        corrupted[mask] = random_tokens[mask]
        return corrupted
    if mode == "random_noise":
        return torch.randint(0, vocab_size, clean_tokens.shape, device=clean_tokens.device, dtype=clean_tokens.dtype, generator=generator)
    raise ValueError(f"Unknown condition: {mode}")


def extract_pre_acts(encode_out):
    if torch.is_tensor(encode_out):
        return encode_out
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    if isinstance(encode_out, (list, tuple)) and encode_out:
        return encode_out[0]
    raise TypeError(f"Unsupported SAE encode output type: {type(encode_out)}")


def apply_topk_dense(activations: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, activations.shape[-1])
    vals, inds = torch.topk(activations, k=k, dim=-1)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, inds, vals)
    return sparse, sparse != 0


def get_topk_sparse(activations: torch.Tensor, k: int):
    k = min(k, activations.shape[-1])
    vals, inds = torch.topk(activations, k=k, dim=-1)
    vals = torch.relu(vals)
    active = torch.zeros_like(activations, dtype=torch.bool)
    active.scatter_(-1, inds, vals > 0)
    return vals, inds, active


class SAEAdapter:
    def __init__(self, sae, backend: str, device: str):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act):
        return extract_pre_acts(self.sae.encode(flat_act))

    def decode_topk(self, feature_acts, k: int):
        if self.backend == "sparsify":
            vals, inds, active = get_topk_sparse(feature_acts, k)
            return self.sae.decode(vals, inds), active
        sparse, active = apply_topk_dense(feature_acts, k)
        return self.sae.decode(sparse), active

    def get_latent_dim(self, d_model: int, dtype: torch.dtype):
        if self.backend == "sae_lens":
            return self.sae.cfg.d_sae
        dummy = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return self.encode_pre_acts(dummy).shape[-1]


def load_model_and_sae(config: Dict):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"], dtype=dtype)
    model.eval()
    if config["sae_backend"] == "sae_lens":
        sae, _, _ = SAE.from_pretrained(release=config["sae_release"], sae_id=config["sae_id"], device=CONFIG["DEVICE"])
        hook_name = config["sae_id"]
    elif config["sae_backend"] == "sparsify":
        sae = Sae.load_from_hub(config["sae_release"], hookpoint=config["sae_hookpoint"]).to(CONFIG["DEVICE"])
        hook_name = config["hook_name"]
    else:
        raise ValueError(f"Unknown backend: {config['sae_backend']}")
    sae.eval()
    return model, SAEAdapter(sae, config["sae_backend"], CONFIG["DEVICE"]), hook_name, dtype


def exact_log_binom(m: int, P: int) -> float:
    if P < 0 or P > m:
        raise ValueError(f"Invalid P={P} for m={m}")
    if P == 0 or P == m:
        return 0.0
    q = min(P, m - P)
    return math.lgamma(m + 1) - math.lgamma(q + 1) - math.lgamma(m - q + 1)


def build_clean_calibration_pool(model_key, model, sae_adapter, hook_name, d_sae, batch_size):
    print(f"\n{model_key}: building one clean calibration pool G*")
    iterator = build_text_iterator(CONFIG["CALIBRATION_SEED"])
    pool_mask = torch.zeros(d_sae, dtype=torch.bool, device=CONFIG["DEVICE"])
    processed = 0
    pbar = tqdm(total=CONFIG["CALIBRATION_TOKENS"], desc=f"{model_key} | clean calibration")
    while processed < CONFIG["CALIBRATION_TOKENS"]:
        tokens = get_clean_text_batch(model, batch_size, iterator)
        if tokens is None:
            break
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name]
            feat = sae_adapter.encode_pre_acts(act.reshape(-1, act.shape[-1]))
            _, active = sae_adapter.decode_topk(feat, CONFIG["TOP_K"])
            pool_mask |= active.any(dim=0)
        n_tok = tokens.numel()
        processed += n_tok
        pbar.update(n_tok)
    pbar.close()
    P = int(pool_mask.sum().item())
    print(f"{model_key}: calibration tokens={processed:,}, P={P:,}/{d_sae:,} ({P/d_sae:.6f})")
    return pool_mask


def evaluate_condition(model_key, condition_cfg, model, sae_adapter, hook_name, pool_mask, vocab_size, batch_size):
    mode = condition_cfg["mode"]
    iterator = build_text_iterator(CONFIG["EVALUATION_STREAM_SEED"])
    generator = None
    if mode == "corrupted_english":
        generator = make_generator(CONFIG["CORRUPTION_SEED"])
    elif mode == "random_noise":
        generator = make_generator(CONFIG["RANDOM_NOISE_SEED"])

    metrics = {"loss_M": [], "loss_h_G": [], "epsilon_loss": [], "pool_violation": []}
    tokens_processed = 0
    n_sequences = 0
    pbar = tqdm(total=CONFIG["TEST_TOKENS"], desc=f"{model_key} | {condition_cfg['short_name']} | eval")

    while tokens_processed < CONFIG["TEST_TOKENS"]:
        clean_tokens = get_clean_text_batch(model, batch_size, iterator)
        if clean_tokens is None:
            break
        tokens = apply_evaluation_condition(clean_tokens, mode, vocab_size, generator)

        with torch.no_grad():
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            loss_M = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

            act = cache[hook_name]
            feat = sae_adapter.encode_pre_acts(act.reshape(-1, act.shape[-1]))

            recons_u, active_u = sae_adapter.decode_topk(feat, CONFIG["TOP_K"])
            recons_u = recons_u.reshape(act.shape).to(act.dtype)

            masked = feat * pool_mask.unsqueeze(0).to(feat.dtype)
            recons_r, _ = sae_adapter.decode_topk(masked, CONFIG["TOP_K"])
            recons_r = recons_r.reshape(act.shape).to(act.dtype)

            violation = active_u & (~pool_mask.unsqueeze(0))
            seq_violation = violation.any(dim=-1).reshape(tokens.shape[0], -1).any(dim=-1).float()

            def hook_u(activations, hook):
                return recons_u

            logits_u = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_u)])
            loss_u = smoothed_bpd_loss(logits_u, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

            def hook_r(activations, hook):
                return recons_r

            logits_r = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_r)])
            loss_r = smoothed_bpd_loss(logits_r, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

            gap = torch.abs(loss_M - loss_u)
            metrics["loss_M"].extend(loss_M.detach().cpu().float().numpy())
            metrics["loss_h_G"].extend(loss_r.detach().cpu().float().numpy())
            metrics["epsilon_loss"].extend(gap.detach().cpu().float().numpy())
            metrics["pool_violation"].extend(seq_violation.detach().cpu().float().numpy())

        n_tok = tokens.numel()
        tokens_processed += n_tok
        n_sequences += tokens.shape[0]
        pbar.update(n_tok)

    pbar.close()
    return {
        "Condition": condition_cfg["name"],
        "Condition Short": condition_cfg["short_name"],
        "N": int(n_sequences),
        "Evaluation Tokens": int(tokens_processed),
        "R_hat_M": float(np.mean(metrics["loss_M"])),
        "R_hat_hG": float(np.mean(metrics["loss_h_G"])),
        "eps_loss_hat": float(np.mean(metrics["epsilon_loss"])),
        "eta_hat": float(np.mean(metrics["pool_violation"])),
    }


def add_certificate_terms(result: Dict, vocab_size: int, m: int, P: int):
    alpha = CONFIG["ALPHA"]
    delta = CONFIG["DELTA_CONF"]
    N = result["N"]

    B = math.log2(vocab_size / alpha)
    Delta = math.log2(1.0 + ((1.0 - alpha) * vocab_size) / alpha)
    baseline = math.log2(vocab_size)
    log_C = exact_log_binom(m, P)

    mismatch_emp = Delta * result["eta_hat"]
    proxy_conc = Delta * math.sqrt(math.log(3.0 / delta) / (2.0 * N))
    eta_conc = Delta * math.sqrt(math.log(3.0 / delta) / (2.0 * N))
    eps_conc = Delta * math.sqrt(math.log(6.0 / delta) / (2.0 * N))
    occam_proxy_conc = Delta * math.sqrt((log_C + math.log(3.0 / delta)) / (2.0 * N))

    U_SAE = result["R_hat_hG"] + result["eps_loss_hat"] + mismatch_emp + proxy_conc + eta_conc + eps_conc
    U_occam = result["R_hat_hG"] + result["eps_loss_hat"] + mismatch_emp + eta_conc + occam_proxy_conc + eps_conc

    direct_M = result["R_hat_M"] + Delta * math.sqrt(math.log(1.0 / delta) / (2.0 * N))
    direct_hG = result["R_hat_hG"] + Delta * math.sqrt(math.log(1.0 / delta) / (2.0 * N))

    result.update({
        "m": int(m), "P": int(P), "P_over_m": P / m,
        "B": B, "Delta": Delta, "Uniform_Baseline": baseline,
        "exact_log_C_m_P": log_C,
        "Pool_Mismatch_Delta_eta": mismatch_emp,
        "Fixed_Proxy_Concentration": proxy_conc,
        "Eta_Concentration": eta_conc,
        "Reconstruction_Concentration": eps_conc,
        "Exact_P_Proxy_Concentration": occam_proxy_conc,
        "Direct_U_M": direct_M, "Direct_U_hG": direct_hG,
        "U_SAE": U_SAE, "U_SAE_Occam": U_occam,
        "Occam_Increment": U_occam - U_SAE,
        "U_SAE_Nonvacuous": U_SAE < baseline,
        "U_SAE_Occam_Nonvacuous": U_occam < baseline,
    })
    return result


def run_decomposition_for_model(model_key: str, model_cfg: Dict) -> pd.DataFrame:
    print(f"\n{'='*80}\nRunning revised corruption decomposition for {model_key}\n{'='*80}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model, sae_adapter, hook_name, dtype = load_model_and_sae(model_cfg)
    vocab_size = model.cfg.d_vocab
    d_sae = sae_adapter.get_latent_dim(model.cfg.d_model, dtype)
    batch_size = model_cfg.get("batch_size", CONFIG["BATCH_SIZE_DEFAULT"])

    pool_mask = build_clean_calibration_pool(model_key, model, sae_adapter, hook_name, d_sae, batch_size)
    P = int(pool_mask.sum().item())
    print(f"{model_key}: fixed G* across all conditions | m={d_sae:,}, P={P:,}, P/m={P/d_sae:.6f}, lnC={exact_log_binom(d_sae,P):.3f}")

    rows: List[Dict] = []
    for cond_cfg in CONFIG["CONDITIONS"]:
        r = evaluate_condition(model_key, cond_cfg, model, sae_adapter, hook_name, pool_mask, vocab_size, batch_size)
        r["Model"] = model_key
        r = add_certificate_terms(r, vocab_size, d_sae, P)
        rows.append(r)
        print(f"{model_key} | {r['Condition Short']}: R(hG*)={r['R_hat_hG']:.3f} | eps={r['eps_loss_hat']:.3f} | eta={r['eta_hat']:.4f} | U_SAE={r['U_SAE']:.3f} | U_SAE+Occam={r['U_SAE_Occam']:.3f} | baseline={r['Uniform_Baseline']:.3f}")

    assert len({r["P"] for r in rows}) == 1
    assert len({round(r["exact_log_C_m_P"], 10) for r in rows}) == 1

    del model, sae_adapter, pool_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def plot_decomposition(df: pd.DataFrame):
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    comps = ["R_hat_hG", "eps_loss_hat", "Pool_Mismatch_Delta_eta"]
    labels = {
        "R_hat_hG": r"Empirical proxy risk $\widehat R(h_{G^\star})$",
        "eps_loss_hat": r"Reconstruction gap $\widehat\epsilon_{\rm loss}$",
        "Pool_Mismatch_Delta_eta": r"Pool mismatch $\Delta\widehat\eta$",
    }
    colors = {"R_hat_hG": "#4C78A8", "eps_loss_hat": "#F58518", "Pool_Mismatch_Delta_eta": "#E45756"}
    condition_order = [c["name"] for c in CONFIG["CONDITIONS"]]
    short_names = {c["name"]: c["short_name"] for c in CONFIG["CONDITIONS"]}
    model_order = [m for m in CONFIG["MODELS"] if m in df["Model"].unique()]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(model_order), figsize=(5.3 * len(model_order), 5.0), sharey=False)
    if len(model_order) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_order):
        sub = df[df["Model"] == model_name].copy()
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=condition_order, ordered=True)
        sub = sub.sort_values("Condition")
        x = np.arange(len(sub)); width = 0.58; bottoms = np.zeros(len(sub))

        for comp in comps:
            vals = sub[comp].to_numpy()
            ax.bar(x, vals, bottom=bottoms, width=width, color=colors[comp], edgecolor="white", linewidth=0.8, zorder=3)
            bottoms += vals

        ax.plot(x, sub["U_SAE"], linestyle="none", marker="o", markersize=7, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        ax.plot(x, sub["U_SAE_Occam"], linestyle="none", marker="D", markersize=6.5, markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.2, zorder=6)

        baseline = float(sub["Uniform_Baseline"].iloc[0])
        ax.axhline(baseline, color="black", linestyle=(0, (5, 2)), linewidth=1.4, zorder=4)

        for xi, u1, u2 in zip(x, sub["U_SAE"], sub["U_SAE_Occam"]):
            ax.text(xi - 0.03, u1 + 0.25, f"{u1:.1f}", ha="right", va="bottom", fontsize=8.5)
            ax.text(xi + 0.03, u2 + 0.25, f"{u2:.1f}", ha="left", va="bottom", fontsize=8.5, color="#555555")

        P = int(sub["P"].iloc[0]); ratio = float(sub["P_over_m"].iloc[0]); logC = float(sub["exact_log_C_m_P"].iloc[0])
        ax.text(0.5, 0.98, rf"fixed $P={P:,}$, $P/m={ratio:.4f}$" + "\n" + rf"$\ln\binom{{m}}{{P}}={logC:.1f}$ nats", transform=ax.transAxes, ha="center", va="top", fontsize=8.7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB", alpha=0.9))

        ax.set_xticks(x)
        ax.set_xticklabels([short_names[c] for c in sub["Condition"].tolist()])
        ax.set_title(model_name, fontweight="bold")
        ax.grid(axis="y", linestyle="-", alpha=0.16, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_axisbelow(True)
        ymax = max(float(sub["U_SAE_Occam"].max()), float(sub["U_SAE"].max()), baseline) * 1.20
        ax.set_ylim(0, ymax)

    axes[0].set_ylabel("Certificate / loss contribution (bits)", fontweight="bold")
    fig.supxlabel("Evaluation distribution", y=0.035, fontweight="bold")
    fig.suptitle(f"Certificate Decomposition under Synthetic Corruption (Top-k={CONFIG['TOP_K']})", fontsize=16, fontweight="bold", y=1.02)

    legend = [Patch(facecolor=colors[c], label=labels[c]) for c in comps]
    legend += [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linestyle="none", label=r"$U_{\rm SAE}$"),
        Line2D([0], [0], marker="D", color="#555555", markerfacecolor="white", linestyle="none", label=r"$U_{\rm SAE+Occam}$"),
        Line2D([0], [0], color="black", linestyle=(0, (5, 2)), label=r"$R_{\rm unif}$"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=6, frameon=False)
    plt.tight_layout(rect=[0.02, 0.07, 0.98, 0.86])

    png_path = os.path.join(CONFIG["OUTPUT_DIR"], "certificate_decomposition_synthetic_corruption_updated.png")
    pdf_path = os.path.join(CONFIG["OUTPUT_DIR"], "certificate_decomposition_synthetic_corruption_updated.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}\nSaved: {pdf_path}")


def main():
    all_results = []
    for model_key, model_cfg in CONFIG["MODELS"].items():
        try:
            all_results.append(run_decomposition_for_model(model_key, model_cfg))
        except Exception as exc:
            print(f"Skipping {model_key} due to error: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError("No model completed successfully.")

    df = pd.concat(all_results, ignore_index=True)
    print("\nFinal revised results:\n")
    print(df.to_string(index=False))

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "certificate_decomposition_synthetic_corruption_updated.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    plot_decomposition(df)


if __name__ == "__main__":
    main()
